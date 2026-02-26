import AVFoundation

final class CameraSessionController {
    let session = AVCaptureSession()

    private let sessionQueue = DispatchQueue(label: "camera.session.queue")
    private var isConfigured = false
    private var isConfiguring = false
    private var isSessionRunningDesired = false

    func start(videoOutput: AVCaptureVideoDataOutput) {
        sessionQueue.async {
            self.isSessionRunningDesired = true
            self.configureIfNeeded(videoOutput: videoOutput)
            self.applyRunStateIfPossible()
        }
    }

    func stop() {
        sessionQueue.async {
            self.isSessionRunningDesired = false
            self.applyRunStateIfPossible()
        }
    }

    private func configureIfNeeded(videoOutput: AVCaptureVideoDataOutput) {
        dispatchPrecondition(condition: .onQueue(sessionQueue))

        guard !isConfigured else { return }
        guard !isConfiguring else { return }
        isConfiguring = true

        session.beginConfiguration()
        var configured = false

        defer {
            session.commitConfiguration()
            isConfiguring = false
            if configured {
                isConfigured = true
            }
        }

        if session.canSetSessionPreset(.hd1280x720) {
            session.sessionPreset = .hd1280x720
        } else {
            session.sessionPreset = .high
        }

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            return
        }

        guard let input = try? AVCaptureDeviceInput(device: device), session.canAddInput(input) else {
            return
        }
        session.addInput(input)

        guard session.canAddOutput(videoOutput) else {
            return
        }
        session.addOutput(videoOutput)

        if let connection = videoOutput.connection(with: .video) {
            if connection.isVideoOrientationSupported {
                connection.videoOrientation = .portrait
            }
            if connection.isVideoStabilizationSupported {
                connection.preferredVideoStabilizationMode = .off
            }
        }

        configured = true
    }

    private func applyRunStateIfPossible() {
        dispatchPrecondition(condition: .onQueue(sessionQueue))

        guard isConfigured else { return }
        guard !isConfiguring else { return }

        if isSessionRunningDesired {
            guard !session.isRunning else { return }
            session.startRunning()
        } else {
            guard session.isRunning else { return }
            session.stopRunning()
        }
    }
}
