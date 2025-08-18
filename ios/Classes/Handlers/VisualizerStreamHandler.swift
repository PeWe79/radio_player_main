/*
 * VisualizerStreamHandler.swift
 *
 * Copyright (c) 2020-2025 Ilia Chirkunov <contact@cheebeez.com>
 *
 * This source code is licensed under the CC BY-NC-SA 4.0.
 * See https://creativecommons.org/licenses/by-nc-sa/4.0/
 */

import Flutter

/// Handles the event stream for visualizer data updates to Flutter.
class VisualizerStreamHandler: NSObject, FlutterStreamHandler, RadioPlayerVisualizerDelegate {
    private var eventSink: FlutterEventSink?
    private weak var playerService: RadioPlayerService?

    /// Initializes the stream handler with a player service instance.
    init(playerService: RadioPlayerService) {
        self.playerService = playerService
        super.init()
    }

    /// Called when Flutter starts listening to the event stream.
    func onListen(withArguments arguments: Any?, eventSink events: @escaping FlutterEventSink) -> FlutterError? {
        eventSink = events
        playerService?.visualizerDelegate = self
        return nil
    }

    /// Called when Flutter stops listening to the event stream.
    func onCancel(withArguments arguments: Any?) -> FlutterError? {
        eventSink = nil
        if playerService?.visualizerDelegate === self {
            playerService?.visualizerDelegate = nil
        }
        return nil
    }

    /// Relays FFT data from the player service to Flutter.
    func didProcessFft(bands: [Int]) {
        print("VisualizerStreamHandler sending to Flutter: \(bands)")
        
        DispatchQueue.main.async {
            self.eventSink?(bands)
        }
    }
}