/*
 * FftAudioProcessor.kt
 *
 * Copyright (c) 2020-2025 Ilia Chirkunov
 *
 * This source code is licensed under the CC BY-NC-SA 4.0.
 * See https://creativecommons.org/licenses/by-nc-sa/4.0/
 */

package com.cheebeez.radio_player

import android.os.SystemClock
import androidx.media3.common.C
import androidx.media3.common.audio.AudioProcessor
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import org.jtransforms.fft.DoubleFFT_1D
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.*

/// Processes a real-time audio stream to generate data for a visualizer.
class FftAudioProcessor(private val coroutineScope: CoroutineScope) : AudioProcessor {

    companion object {
        // FFT size 2¹⁰ = 1024
        private const val FFT_SIZE = 1024
        private const val HOP_SIZE = FFT_SIZE / 2

        // Visualizer configuration
        private const val BANDS_COUNT = 16
        private const val UPDATE_INTERVAL_MS: Long = 100

        // Frequency range for visualization
        private const val LOW_CUT_HZ = 50.0
        private const val HIGH_CUT_HZ = 8000.0

        // Dynamic range for mapping to 0…255
        private const val DYNAMIC_RANGE_DB = 72.0

        // Pre-emphasis to counter pink spectrum 1∕f: w = f^α
        private const val PREEMPHASIS_POWER = 0.1

        // Temporal smoothing per band 0…1 EMA
        private const val ALPHA_ATTACK = 0.6
        private const val ALPHA_RELEASE = 0.15

        private const val POWER_EPS = 1e-12
    }

    var isEnabled: Boolean = false

    // Audio format
    private var sampleRateHz: Int = 44100
    private var channelCount: Int = 0
    private var encoding: Int = C.ENCODING_PCM_16BIT

    // Throttling for UI updates
    private var lastUpdateTime: Long = 0

    // FFT instance used only on the worker thread
    private val fft = DoubleFFT_1D(FFT_SIZE.toLong())

    // Pass-through to downstream
    private var outputBuffer: ByteBuffer = AudioProcessor.EMPTY_BUFFER
    private var streamEnded = false

    // Worker to serialize heavy FFT + post-processing
    // It now accepts raw ByteBuffers to process off the audio thread.
    private var byteChannel = Channel<ByteBuffer>(capacity = Channel.CONFLATED)
    private lateinit var workerJob: Job

    // Derived DSP state used on the worker thread
    private val window = DoubleArray(FFT_SIZE)
    private val bandEdges = IntArray(BANDS_COUNT + 1)
    private val binWeights = DoubleArray(FFT_SIZE / 2)
    private val smoothedBands = DoubleArray(BANDS_COUNT)
    private var powerScale = 1.0

    // Guards EMA state for future multi-thread usage
    private val processingMutex = Mutex()

    // Flag to rebuild DSP state on the worker thread after format changes
    @Volatile
    private var needsRebuild = true

    init {
        // Build initial state
        rebuildDerivedState()

        // Start worker coroutine to process incoming FFT blocks
        startWorker()
    }

    /// Starts the worker coroutine that processes audio blocks.
    private fun startWorker() {
        workerJob = coroutineScope.launch(Dispatchers.Default) {
            // Reusable buffers and state are now private to the worker to avoid races
            val blockBuffer = DoubleArray(FFT_SIZE)
            var blockIndex = 0
            val fftWorkBuffer = DoubleArray(FFT_SIZE)
            val powerSpectrumBuffer = DoubleArray(FFT_SIZE / 2)

            // The worker now consumes raw ByteBuffers from the audio thread
            for (inputBytes in byteChannel) {
                if (!isActive) break

                if (needsRebuild) {
                    rebuildDerivedState()
                    needsRebuild = false
                }

                // All heavy processing is now done here, off the audio thread.
                // This includes sample conversion and block building.
                val bytesPerSample = if (encoding == C.ENCODING_PCM_FLOAT) 4 else 2
                val bytesPerFrame = bytesPerSample * channelCount

                while (inputBytes.remaining() >= bytesPerFrame) {
                    var acc = 0.0
                    if (encoding == C.ENCODING_PCM_FLOAT) {
                        for (ch in 0 until channelCount) {
                            acc += inputBytes.float.toDouble()
                        }
                    } else {
                        for (ch in 0 until channelCount) {
                            // Normalize to −1…1
                            acc += inputBytes.short.toDouble() / 32768.0
                        }
                    }
                    val mono = acc / channelCount.toDouble()

                    // Add sample to the current block
                    blockBuffer[blockIndex] = mono
                    blockIndex += 1

                    // When a block is full, process it
                    if (blockIndex >= FFT_SIZE) {
                        // Apply Hann window: w[n] = 0.5 × 1 − 0.5 × cos 2πn ÷ N−1
                        for (i in 0 until FFT_SIZE) {
                            fftWorkBuffer[i] = blockBuffer[i] * window[i]
                        }

                        // In-place real FFT
                        fft.realForward(fftWorkBuffer)

                        // Compute power spectrum for half-spectrum without Nyquist bin:
                        // P₀ = Re₀²
                        // Pₖ = Reₖ² + Imₖ² for k = 1…N∕2−1 where Reₖ = a[2k], Imₖ = a[2k+1]
                        val n2 = FFT_SIZE / 2
                        powerSpectrumBuffer[0] = fftWorkBuffer[0] * fftWorkBuffer[0]
                        for (k in 1 until n2) {
                            val re = fftWorkBuffer[2 * k]
                            val im = fftWorkBuffer[2 * k + 1]
                            powerSpectrumBuffer[k] = re * re + im * im
                        }

                        // Normalize power to be dBFS-like
                        // Scale DC with 0.5 to account for single-sided spectrum energy
                        if (n2 > 0) {
                            powerSpectrumBuffer[0] *= powerScale * 0.5
                            val scale = powerScale
                            for (k in 1 until n2) {
                                powerSpectrumBuffer[k] *= scale
                            }
                        }

                        // Apply pre-emphasis per bin: wₖ = 0 for k=0, else wₖ = f^α normalized to fHigh
                        for (k in 0 until n2) {
                            powerSpectrumBuffer[k] *= binWeights[k]
                        }

                        // Aggregate into log-spaced bands with EMA smoothing and map to 0…255
                        val bands = bandsFromPower(powerSpectrumBuffer)
                        if (bands.isNotEmpty()) {
                            // Throttle UI updates on the main thread
                            withContext(Dispatchers.Main) {
                                val now = SystemClock.uptimeMillis()
                                if (now - lastUpdateTime >= UPDATE_INTERVAL_MS) {
                                    lastUpdateTime = now
                                    VisualizerEventsController.sendData(bands.toList())
                                }
                            }
                        }

                        // Implement 50% overlap: shift the last half to the beginning
                        System.arraycopy(blockBuffer, FFT_SIZE - HOP_SIZE, blockBuffer, 0, HOP_SIZE)
                        blockIndex = HOP_SIZE
                    }
                }
            }
        }
    }

    /// Configures the processor based on the incoming audio format.
    override fun configure(inputAudioFormat: AudioProcessor.AudioFormat): AudioProcessor.AudioFormat {
        if (inputAudioFormat.encoding != C.ENCODING_PCM_16BIT && inputAudioFormat.encoding != C.ENCODING_PCM_FLOAT) {
            throw AudioProcessor.UnhandledAudioFormatException(inputAudioFormat)
        }
        encoding = inputAudioFormat.encoding

        val newRate = inputAudioFormat.sampleRate
        val newCh = inputAudioFormat.channelCount.coerceAtLeast(1)

        // Only mark for rebuild; actual rebuild happens on the worker thread
        if (newRate > 0 && newRate != sampleRateHz) {
            sampleRateHz = newRate
            needsRebuild = true
        }
        if (newCh != channelCount) {
            channelCount = newCh
        }

        return inputAudioFormat
    }

    /// Queues a block of audio data for processing and forwards it downstream unchanged.
    override fun queueInput(buffer: ByteBuffer) {
        // Create a read-only view for analysis before passing the original buffer on.
        val analysisView = buffer.asReadOnlyBuffer().order(ByteOrder.LITTLE_ENDIAN)

        // Pass the original buffer downstream for playback. DO NOT modify its state.
        // The player expects to receive it exactly as it was sent.
        outputBuffer = buffer

        // If disabled or format incomplete or no data, skip analysis
        if (!isEnabled || channelCount <= 0 || !analysisView.hasRemaining()) return

        // To safely pass the buffer to another thread, we must make a copy,
        // as the original buffer is owned by Media3 and might be recycled.
        val bufferCopy = ByteBuffer.allocateDirect(analysisView.remaining())
            .order(ByteOrder.LITTLE_ENDIAN)
        bufferCopy.put(analysisView)
        bufferCopy.flip()

        // Non-blocking send; if the worker is backed up, it's okay to drop frames.
        // This is the ONLY work done on the audio thread. It's very fast.
        byteChannel.trySend(bufferCopy)
    }

    /// Converts power spectrum into BANDS_COUNT integer magnitudes 0…255 using log-spaced bands and EMA smoothing.
    private suspend fun bandsFromPower(power: DoubleArray): IntArray {
        val n2 = FFT_SIZE / 2
        if (bandEdges.size != BANDS_COUNT + 1 || power.size != n2) return IntArray(0)

        val out = IntArray(BANDS_COUNT)

        // Protect EMA state
        processingMutex.withLock {
            for (i in 0 until BANDS_COUNT) {
                val s = bandEdges[i].coerceIn(0, n2)
                val e = bandEdges[i + 1].coerceIn(s + 1, n2)

                var sumP = 0.0
                for (k in s until e) sumP += power[k]

                // dB = 10 × log10 max P, ε
                val dB = 10.0 * log10(max(sumP, POWER_EPS))

                // Scale to 0…1 within dynamic range
                var scaled = (dB + DYNAMIC_RANGE_DB) / DYNAMIC_RANGE_DB
                if (!scaled.isFinite()) scaled = 0.0
                scaled = scaled.coerceIn(0.0, 1.0)

                // EMA smoothing with separate attack and release
                val prev = smoothedBands[i]
                val alpha = if (scaled > prev) ALPHA_ATTACK else ALPHA_RELEASE
                val smoothed = prev + alpha * (scaled - prev)
                smoothedBands[i] = smoothed

                // Map to 0…255
                out[i] = (smoothed * 255.0).roundToInt().coerceIn(0, 255)
            }
        }
        return out
    }

    /// Rebuilds log-spaced band edges, bin weights and normalization based on current sample rate.
    /// Called only on the worker thread to avoid data races.
    private fun rebuildDerivedState() {
        val sr = max(sampleRateHz, 1)
        val ny = sr / 2.0
        val n2 = FFT_SIZE / 2

        // Build Hann window and compute its average power U = Σ w² ÷ N
        var winPow = 0.0
        for (n in 0 until FFT_SIZE) {
            // Hann: w[n] = 0.5 × 1 − 0.5 × cos 2πn ÷ N−1
            val w = 0.5 * (1.0 - cos(2.0 * PI * n / (FFT_SIZE - 1).toDouble()))
            window[n] = w
            winPow += w * w
        }
        val U = max(winPow / FFT_SIZE, 1e-12)
        // powerScale = 2 ÷ N² ÷ U
        powerScale = 2.0 / (FFT_SIZE.toDouble() * FFT_SIZE.toDouble() * U)

        // Frequency bounds
        val fLow = max(LOW_CUT_HZ, sr.toDouble() / FFT_SIZE)
        val fHigh = min(HIGH_CUT_HZ, ny * 0.98)

        // Log-spaced edges in bins
        val logStep = ln(fHigh / fLow) / BANDS_COUNT
        for (i in 0..BANDS_COUNT) {
            val f = fLow * exp(logStep * i)
            var bin = (f * FFT_SIZE / sr).toInt()
            bin = bin.coerceIn(0, n2)
            bandEdges[i] = bin
            if (i > 0 && bandEdges[i] <= bandEdges[i - 1]) {
                bandEdges[i] = (bandEdges[i - 1] + 1).coerceAtMost(n2)
            }
        }
        bandEdges[BANDS_COUNT] = n2

        // Pre-emphasis weights per bin: wₖ = 0 for k=0; else wₖ = min max f ÷ fHigh, 0, 1 ^ α
        for (k in 0 until n2) {
            val f = k.toDouble() * sr / FFT_SIZE
            val norm = (f / fHigh).coerceIn(0.0, 1.0)
            binWeights[k] = if (k == 0) 0.0 else norm.pow(PREEMPHASIS_POWER)
        }
        // Keep smoothedBands as-is for continuity
    }

    /// Clears the internal buffer of any accumulated data and resets stream flags.
    override fun flush() {
        // Drain any pending buffers from the channel
        while (byteChannel.tryReceive().isSuccess) { /* drain */ }
        outputBuffer = AudioProcessor.EMPTY_BUFFER
        streamEnded = false
        // Keep smoothing to preserve continuity across flush
    }

    /// Resets the processor to its initial state.
    override fun reset() {
        flush()

        // Stop the current worker and clean up resources to prevent leaks.
        release()

        // Re-initialize for the next use.
        byteChannel = Channel(Channel.CONFLATED)
        startWorker()

        sampleRateHz = 44100
        channelCount = 0
        encoding = C.ENCODING_PCM_16BIT
        isEnabled = false
        needsRebuild = true
    }

    /// Releases the resources used by the processor, stopping the background worker.
    fun release() {
        byteChannel.close()
        if (::workerJob.isInitialized) {
            workerJob.cancel()
        }
    }

    /// Tells ExoPlayer whether this processor is currently active.
    override fun isActive(): Boolean = isEnabled

    /// Signals that the processor has no more output data to return.
    override fun isEnded(): Boolean {
        return streamEnded && outputBuffer === AudioProcessor.EMPTY_BUFFER
    }

    /// Returns an empty buffer because this processor only analyzes the audio stream.
    override fun getOutput(): ByteBuffer {
        val bufferToReturn = outputBuffer
        outputBuffer = AudioProcessor.EMPTY_BUFFER
        return bufferToReturn
    }

    /// Called when the audio stream ends.
    override fun queueEndOfStream() {
        streamEnded = true
    }
}
