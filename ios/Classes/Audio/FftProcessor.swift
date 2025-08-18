/*
 * FftProcessor.swift
 *
 * Copyright (c) 2020-2025 Ilia Chirkunov
 *
 * This source code is licensed under the CC BY-NC-SA 4.0
 * See https://creativecommons.org/licenses/by-nc-sa/4.0/
 */

import Foundation
import Accelerate
import CoreMedia
import AudioToolbox

/// Processes a real-time audio stream to generate data for a visualizer using FFT.
class FftProcessor {

    // FFT size 2¹⁰ = 1024
    private static let FFT_SIZE: vDSP_Length = 1024
    private static let FFT_LOG2N: vDSP_Length = 10

    // Visualizer configuration
    private static let BANDS_COUNT = 16
    private static let UPDATE_INTERVAL_MS: UInt64 = 100

    // Frequency range for visualization
    private static let LOW_CUT_HZ: Float = 50.0
    private static let HIGH_CUT_HZ: Float = 8000.0

    // Dynamic range for mapping to 0…255
    private static let DYNAMIC_RANGE_DB: Float = 72.0

    // Pre-emphasis to counter pink spectrum 1/f: w = f^α
    private static let PREEMPHASIS_POWER: Float = 0.1

    // Temporal smoothing per band 0…1 (EMA)
    private static let ALPHA_ATTACK: Float = 0.6
    private static let ALPHA_RELEASE: Float = 0.15

    private static let POWER_EPS: Float = 1e-12

    // Throttling
    private var lastUpdateTime: UInt64 = 0

    // FFT
    private let fftSetup: FFTSetup

    // Streaming buffers
    private var inputBuffer = [Float](repeating: 0, count: Int(FFT_SIZE))
    private var inputBufferIndex = 0

    // Precomputed Hann window to reduce spectral leakage
    private let window: [Float]

    // Derived DSP state
    private var sampleRate: Double = 44100.0
    private var bandEdges: [Int] = Array(repeating: 0, count: BANDS_COUNT + 1)
    private var binWeights: [Float] = Array(repeating: 0, count: Int(FFT_SIZE / 2))
    private var needsRebuild = true

    // Smoothing state per band in 0…1
    private var smoothedBands: [Float] = Array(repeating: 0, count: BANDS_COUNT)

    // Window/FFT normalization for power → dBFS-like
    private var powerScale: Float = 1.0

    // Background queue for heavy FFT work
    private let processingQueue = DispatchQueue(label: "com.cheebeez.radio_player.fft", qos: .userInitiated)

    // Reusable buffers to avoid frequent allocations
    private var windowedBuffer = [Float](repeating: 0, count: Int(FFT_SIZE))
    private var realBuffer = [Float](repeating: 0, count: Int(FFT_SIZE / 2))
    private var imagBuffer = [Float](repeating: 0, count: Int(FFT_SIZE / 2))
    private var powerBuffer = [Float](repeating: 0, count: Int(FFT_SIZE / 2))
    private var weightedPowerBuffer = [Float](repeating: 0, count: Int(FFT_SIZE / 2))

    // The delegate to notify with processed FFT data.
    weak var delegate: RadioPlayerVisualizerDelegate?

    /// Initializes the FFT processor. Fails if the vDSP setup cannot be created.
    init?() {
        // Create a reusable setup object for the FFT calculation.
        guard let setup = vDSP_create_fftsetup(FftProcessor.FFT_LOG2N, FFTRadix(kFFTRadix2)) else {
            return nil
        }
        self.fftSetup = setup

        // Build Hann window length N: w[n] = 0.5 × 1 − 0.5 × cos 2πn ÷ N−1
        var w = [Float](repeating: 0, count: Int(FftProcessor.FFT_SIZE))
        vDSP_hann_window(&w, vDSP_Length(w.count), Int32(vDSP_HANN_NORM))
        self.window = w
        
        // Ensure bands/weights are ready even if incoming ASBD has the same SR as default
        rebuildDerivedState()
        needsRebuild = false
    }

    /// Clean up the vDSP setup object to prevent memory leaks.
    deinit {
        vDSP_destroy_fftsetup(fftSetup)
    }

    /// Processes a block of audio data from an AudioBufferList.
    /// This method extracts samples, converts them to mono, and adds them to a buffer for FFT processing.
    public func process(bufferList: UnsafeMutablePointer<AudioBufferList>, frames: UInt32, asbd: AudioStreamBasicDescription) {
        let frameCount = Int(frames)
        let channels = max(Int(asbd.mChannelsPerFrame), 1)
        let isFloat = (asbd.mFormatFlags & kAudioFormatFlagIsFloat) != 0
        let isNonInterleaved = (asbd.mFormatFlags & kAudioFormatFlagIsNonInterleaved) != 0

        // Check if the sample rate has changed and rebuild derived state if necessary.
        let sr = asbd.mSampleRate
        if sr > 0 {
            if needsRebuild || sr != sampleRate {
                sampleRate = sr
                processingQueue.async { [weak self] in
                    self?.rebuildDerivedState()
                }
                needsRebuild = false
            }
        } else if needsRebuild {
            processingQueue.async { [weak self] in
                self?.rebuildDerivedState()
            }
            needsRebuild = false
        }

        let abl = UnsafeMutableAudioBufferListPointer(bufferList)
        guard abl.count > 0 else { return }

        // Handle different audio formats (non-interleaved vs. interleaved)
        if isNonInterleaved {
            if isFloat {
                // Get pointers to each channel's data plane for float format.
                let planes: [UnsafePointer<Float32>] = (0..<min(channels, abl.count)).compactMap {
                    guard let p = abl[$0].mData?.assumingMemoryBound(to: Float32.self) else { return nil }
                    return UnsafePointer<Float32>(p)
                }
                let planeCount = planes.count
                guard planeCount > 0 else { return }

                // Average all channels to create a mono signal.
                for i in 0..<frameCount {
                    var acc: Float = 0.0
                    for ch in 0..<planeCount { acc += planes[ch][i] }
                    addSampleToBuffer(acc / Float(planeCount))
                }
            } else {
                // Get pointers to each channel's data plane for Int16 format.
                let planes: [UnsafePointer<Int16>] = (0..<min(channels, abl.count)).compactMap {
                    guard let p = abl[$0].mData?.assumingMemoryBound(to: Int16.self) else { return nil }
                    return UnsafePointer<Int16>(p)
                }
                let planeCount = planes.count
                guard planeCount > 0 else { return }

                // Average all channels and normalize to -1.0…1.0.
                for i in 0..<frameCount {
                    var acc: Float = 0.0
                    for ch in 0..<planeCount { acc += Float(planes[ch][i]) / 32768.0 }
                    addSampleToBuffer(acc / Float(planeCount))
                }
            }
        } else {
            // Handle interleaved audio data.
            guard let dataPtr = abl[0].mData else { return }
            if isFloat {
                let pcm = dataPtr.assumingMemoryBound(to: Float32.self)
                // Average all channels for each frame.
                for i in 0..<frameCount {
                    var acc: Float = 0.0
                    let base = i * channels
                    for ch in 0..<channels { acc += pcm[base + ch] }
                    addSampleToBuffer(acc / Float(channels))
                }
            } else {
                let pcm = dataPtr.assumingMemoryBound(to: Int16.self)
                // Average all channels for each frame and normalize.
                for i in 0..<frameCount {
                    var acc: Float = 0.0
                    let base = i * channels
                    for ch in 0..<channels { acc += Float(pcm[base + ch]) / 32768.0 }
                    addSampleToBuffer(acc / Float(channels))
                }
            }
        }
    }

    /// Appends a single mono sample to the input buffer and triggers FFT processing when the buffer is full.
    private func addSampleToBuffer(_ sample: Float) {
        inputBuffer[inputBufferIndex] = sample
        inputBufferIndex += 1

        // When the buffer is full, process the block.
        if inputBufferIndex >= FftProcessor.FFT_SIZE {
            let blockToProcess = inputBuffer

            // Implement a "sliding window" by shifting the last half of the buffer
            // to the beginning, creating a 50% overlap. This provides smoother visualization.
            let overlap = Int(FftProcessor.FFT_SIZE / 2)
            inputBuffer.replaceSubrange(0..<overlap, with: inputBuffer[Int(FftProcessor.FFT_SIZE) - overlap..<Int(FftProcessor.FFT_SIZE)])
            inputBufferIndex = overlap
            
            // Throttle updates before heavy work; drop blocks if faster than UPDATE_INTERVAL_MS.
            let nowMs = DispatchTime.now().uptimeNanoseconds / 1_000_000
            if nowMs - lastUpdateTime >= FftProcessor.UPDATE_INTERVAL_MS {
                lastUpdateTime = nowMs

                // Offload FFT + post-processing to a background thread.
                processingQueue.async { [weak self] in
                    self?.performFft(on: blockToProcess)
                }
            }
        }
    }

    /// Performs the FFT and all subsequent post-processing on a block of audio data.
    private func performFft(on block: [Float]) {
        // Apply Hann window: w[n] * s[n]
        window.withUnsafeBufferPointer { wPtr in
            block.withUnsafeBufferPointer { bPtr in
                vDSP_vmul(bPtr.baseAddress!, 1, wPtr.baseAddress!, 1, &windowedBuffer, 1, vDSP_Length(FftProcessor.FFT_SIZE))
            }
        }

        // Prepare for FFT by converting the real signal into an even/odd split complex format.
        var split = DSPSplitComplex(realp: &realBuffer, imagp: &imagBuffer)
        windowedBuffer.withUnsafeBufferPointer { inPtr in
            inPtr.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: Int(FftProcessor.FFT_SIZE / 2)) { cPtr in
                vDSP_ctoz(cPtr, 2, &split, 1, vDSP_Length(FftProcessor.FFT_SIZE / 2))
            }
        }
        
        // Perform the forward FFT in-place.
        vDSP_fft_zrip(fftSetup, &split, 1, FftProcessor.FFT_LOG2N, FFTDirection(FFT_FORWARD))

        // Compute power spectrum for half-spectrum: Pₖ = Reₖ² + Imₖ²
        powerBuffer[0] = split.realp[0] * split.realp[0] // DC component
        if powerBuffer.count > 1 {
            for k in 1..<powerBuffer.count {
                let re = split.realp[k]
                let im = split.imagp[k]
                powerBuffer[k] = re * re + im * im
            }
        }

        // Normalize power to be dBFS-like.
        let n2 = powerBuffer.count
        if n2 > 0 {
            // Scale DC with 0.5 to account for single-sided spectrum energy.
            powerBuffer[0] *= powerScale * 0.5
            if n2 > 1 {
                // Scale the rest of the spectrum.
                powerBuffer.withUnsafeMutableBufferPointer { ptr in
                    let base = ptr.baseAddress!
                    var scale = powerScale
                    vDSP_vsmul(base.advanced(by: 1), 1, &scale, base.advanced(by: 1), 1, vDSP_Length(n2 - 1))
                }
            }
        }
        
        // Apply pre-emphasis per bin: power[k] * weights[k]
        let finalPower: [Float]
        if binWeights.count == powerBuffer.count {
            powerBuffer.withUnsafeBufferPointer { pPtr in
                binWeights.withUnsafeBufferPointer { wPtr in
                    weightedPowerBuffer.withUnsafeMutableBufferPointer { tPtr in
                        vDSP_vmul(pPtr.baseAddress!, 1, wPtr.baseAddress!, 1, tPtr.baseAddress!, 1, vDSP_Length(powerBuffer.count))
                    }
                }
            }
            finalPower = weightedPowerBuffer
        } else {
            finalPower = powerBuffer
        }
        
        // Aggregate into log-spaced bands with EMA smoothing and map to 0…255.
        let bands = bandsFromPower(finalPower)

        // Dispatch to the main thread for UI safety.
        if !bands.isEmpty {
            DispatchQueue.main.async { [weak self] in
                self?.delegate?.didProcessFft(bands: bands)
            }
        }
    }

    /// Converts power spectrum into BANDS_COUNT integer magnitudes 0…255 using log-spaced bands and EMA smoothing.
    private func bandsFromPower(_ power: [Float]) -> [Int] {
        let n2 = Int(FftProcessor.FFT_SIZE / 2)
        guard bandEdges.count == FftProcessor.BANDS_COUNT + 1, n2 == power.count else { return [] }

        var out = [Int](repeating: 0, count: FftProcessor.BANDS_COUNT)

        for i in 0..<FftProcessor.BANDS_COUNT {
            let s = max(0, min(bandEdges[i], n2))
            let e = max(s + 1, min(bandEdges[i + 1], n2))
            
            // Sum the power within the frequency band.
            var sumP: Float = 0.0
            if e > s {
                for k in s..<e { sumP += power[k] }
            }
            
            // dB = 10 × log10(max(P, ε))
            let dB = 10.0 * log10f(max(sumP, FftProcessor.POWER_EPS))
            
            // Scale to 0…1 within the dynamic range.
            var scaled = (dB + FftProcessor.DYNAMIC_RANGE_DB) / FftProcessor.DYNAMIC_RANGE_DB
            if !scaled.isFinite { scaled = 0.0 }
            scaled = min(max(scaled, 0.0), 1.0)
            
            // EMA smoothing with separate attack/release.
            let prev = smoothedBands[i]
            let alpha = (scaled > prev) ? FftProcessor.ALPHA_ATTACK : FftProcessor.ALPHA_RELEASE
            let smoothed = prev + alpha * (scaled - prev)
            smoothedBands[i] = smoothed
            
            // Map to 0…255.
            out[i] = Int(min(max(smoothed * 255.0, 0.0), 255.0))
        }

        return out
    }

    /// Rebuilds log-spaced band edges, bin weights, and normalization based on the current sample rate.
    private func rebuildDerivedState() {
        let sr = max(sampleRate, 1.0)
        let ny = sr / 2.0
        let n2 = Int(FftProcessor.FFT_SIZE / 2)

        // Frequency bounds.
        let fLow = max(Double(FftProcessor.LOW_CUT_HZ), sr / Double(FftProcessor.FFT_SIZE))
        let fHigh = min(Double(FftProcessor.HIGH_CUT_HZ), ny * 0.98)

        // Create logarithmically spaced band edges in terms of FFT bin indices.
        var edges = [Int](repeating: 0, count: FftProcessor.BANDS_COUNT + 1)
        let logStep = log(fHigh / fLow) / Double(FftProcessor.BANDS_COUNT)
        for i in 0...FftProcessor.BANDS_COUNT {
            let f = fLow * exp(logStep * Double(i))
            var bin = Int(f * Double(FftProcessor.FFT_SIZE) / sr)
            bin = max(0, min(bin, n2))
            edges[i] = bin
            // Ensure each band has at least one bin.
            if i > 0 && edges[i] <= edges[i - 1] {
                edges[i] = min(edges[i - 1] + 1, n2)
            }
        }
        edges[FftProcessor.BANDS_COUNT] = n2
        bandEdges = edges

        // Create pre-emphasis weights per bin: wₖ = (f ÷ fHigh)^α
        var weights = [Float](repeating: 0, count: n2)
        for k in 0..<n2 {
            let f = Double(k) * sr / Double(FftProcessor.FFT_SIZE)
            let norm = min(max(f / fHigh, 0.0), 1.0)
            weights[k] = (k == 0) ? 0.0 : pow(Float(norm), FftProcessor.PREEMPHASIS_POWER)
        }
        binWeights = weights

        // Compute the average power of the window function (U = Σ w² ÷ N) for normalization.
        let N = Float(FftProcessor.FFT_SIZE)
        var winPow: Float = 0
        window.withUnsafeBufferPointer { w in
            vDSP_svesq(w.baseAddress!, 1, &winPow, vDSP_Length(FftProcessor.FFT_SIZE))
        }
        let U = max(winPow / N, 1e-12)

        // powerScale = 2 ÷ (N² × U)
        powerScale = 2.0 / (N * N * U)
    }
}
