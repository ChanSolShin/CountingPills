import AVFoundation
import UIKit

final class CameraCaptureViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    var onResult: ((PillInferenceResult) -> Void)?
    var onProcessingChange: ((Bool) -> Void)?
    var onCaptureStateChange: ((Bool) -> Void)?

    private let session = AVCaptureSession()
    private let previewLayer = AVCaptureVideoPreviewLayer()
    private let videoOutput = AVCaptureVideoDataOutput()

    private let inferenceQueue = DispatchQueue(label: "pill.inference.queue", qos: .userInitiated)
    private let sessionQueue = DispatchQueue(label: "camera.session.queue")
    private let stateQueue = DispatchQueue(label: "camera.state.queue")
    private let videoOutputQueue = DispatchQueue(label: "camera.video.output.queue", qos: .userInitiated)
    private let latestBufferQueue = DispatchQueue(label: "camera.latest.buffer.queue")

    private let framePreparationUseCase: CaptureFramePreparationUseCase
    private let runPillDetectionUseCase: RunPillDetectionUseCase

    private let previewContainerView = UIView()
    private let frameBorderLayer = CAShapeLayer()
    private let dotLayer = CAShapeLayer()
    private let frozenImageView = UIImageView()

    private var captureState = CameraCaptureStateMachine()
    private var isSessionConfigured = false
    private var latestPixelBuffer: CVPixelBuffer?

    init(
        framePreparationUseCase: CaptureFramePreparationUseCase = DefaultCaptureFramePreparationUseCase(
            modelSide: 640,
            variantCount: 8
        ),
        runPillDetectionUseCase: RunPillDetectionUseCase = DefaultRunPillDetectionUseCase()
    ) {
        self.framePreparationUseCase = framePreparationUseCase
        self.runPillDetectionUseCase = runPillDetectionUseCase
        super.init(nibName: nil, bundle: nil)
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        setupViews()
        setupCamera()
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()

        let frameRect = modelFrameRect(in: view.bounds)
        previewContainerView.frame = frameRect

        previewLayer.frame = previewContainerView.bounds
        frozenImageView.frame = previewContainerView.bounds
        dotLayer.frame = previewContainerView.bounds

        frameBorderLayer.frame = previewContainerView.bounds
        frameBorderLayer.path = UIBezierPath(roundedRect: previewContainerView.bounds, cornerRadius: 12).cgPath

        setPreviewPortraitOrientation()
    }

    func captureCurrentFrame() {
        stateQueue.async {
            guard self.captureState.beginCapture() else { return }

            guard
                let sourcePixelBuffer = self.latestBufferQueue.sync(execute: { self.latestPixelBuffer }),
                let ownedPixelBuffer = self.copyPixelBuffer(sourcePixelBuffer)
            else {
                #if DEBUG
                print("[PillDebug] capture-failed: latestPixelBuffer unavailable")
                #endif
                self.finishProcessing(keepCapturedFrame: false)
                return
            }

            DispatchQueue.main.async {
                self.dotLayer.path = nil
                self.onProcessingChange?(true)
            }

            self.inferenceQueue.async { [weak self] in
                self?.processCapturedFrame(ownedPixelBuffer)
            }
        }
    }

    func resumeCameraForRetake() {
        stateQueue.async {
            self.captureState.resetForRetake()
            DispatchQueue.main.async {
                self.frozenImageView.image = nil
                self.frozenImageView.isHidden = true
                self.dotLayer.path = nil
                self.onCaptureStateChange?(false)
                self.onProcessingChange?(false)
            }
        }
    }

    func stopSession() {
        sessionQueue.async {
            if self.session.isRunning {
                self.session.stopRunning()
            }
        }
    }

    private func setupViews() {
        view.backgroundColor = .black

        previewContainerView.clipsToBounds = true
        previewContainerView.layer.cornerRadius = 12
        previewContainerView.backgroundColor = .black
        view.addSubview(previewContainerView)

        previewLayer.videoGravity = .resizeAspectFill
        previewContainerView.layer.addSublayer(previewLayer)

        frozenImageView.contentMode = .scaleToFill
        frozenImageView.backgroundColor = .black
        frozenImageView.isHidden = true
        frozenImageView.clipsToBounds = true
        previewContainerView.addSubview(frozenImageView)

        dotLayer.fillColor = UIColor.systemGreen.cgColor
        dotLayer.strokeColor = UIColor.clear.cgColor
        previewContainerView.layer.addSublayer(dotLayer)

        frameBorderLayer.fillColor = UIColor.clear.cgColor
        frameBorderLayer.strokeColor = UIColor.white.withAlphaComponent(0.9).cgColor
        frameBorderLayer.lineWidth = 2
        previewContainerView.layer.addSublayer(frameBorderLayer)
    }

    private func setupCamera() {
        sessionQueue.async {
            if self.isSessionConfigured {
                if !self.session.isRunning {
                    self.session.startRunning()
                }
                DispatchQueue.main.async {
                    self.previewLayer.session = self.session
                    self.setPreviewPortraitOrientation()
                }
                return
            }

            self.session.beginConfiguration()
            var shouldStartSession = false
            defer {
                self.session.commitConfiguration()

                if shouldStartSession {
                    self.isSessionConfigured = true

                    DispatchQueue.main.async {
                        self.previewLayer.session = self.session
                        self.setPreviewPortraitOrientation()
                    }

                    if !self.session.isRunning {
                        self.session.startRunning()
                    }
                }
            }

            if self.session.canSetSessionPreset(.hd1280x720) {
                self.session.sessionPreset = .hd1280x720
            } else {
                self.session.sessionPreset = .high
            }

            guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
                return
            }

            guard let input = try? AVCaptureDeviceInput(device: device), self.session.canAddInput(input) else {
                return
            }
            self.session.addInput(input)

            self.videoOutput.videoSettings = [
                kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)
            ]
            self.videoOutput.alwaysDiscardsLateVideoFrames = true
            self.videoOutput.setSampleBufferDelegate(self, queue: self.videoOutputQueue)

            guard self.session.canAddOutput(self.videoOutput) else {
                return
            }
            self.session.addOutput(self.videoOutput)

            if let connection = self.videoOutput.connection(with: .video) {
            if connection.isVideoOrientationSupported {
                connection.videoOrientation = .portrait
            }
            if connection.isVideoStabilizationSupported {
                connection.preferredVideoStabilizationMode = .off
            }
            }
            shouldStartSession = true
        }
    }

    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        latestBufferQueue.sync {
            self.latestPixelBuffer = pixelBuffer
        }
    }

    private func processCapturedFrame(_ ownedPixelBuffer: CVPixelBuffer) {
        guard let prepared = autoreleasepool(invoking: { self.framePreparationUseCase.prepare(from: ownedPixelBuffer) }) else {
            finishProcessing(keepCapturedFrame: false)
            return
        }

        stateQueue.async {
            self.captureState.markCapturedFrameShown()
        }

        DispatchQueue.main.async {
            self.frozenImageView.image = prepared.displayImage
            self.frozenImageView.isHidden = false
            self.onCaptureStateChange?(true)
        }

        autoreleasepool {
            defer {
                self.finishProcessing(keepCapturedFrame: true)
            }

            let inferenceResult = self.runPillDetectionUseCase.run(modelPixelBuffers: prepared.modelPixelBuffers)
            let mappedPoints = prepared.mapPointsToDisplay(inferenceResult.points)
            let result = PillInferenceResult(count: inferenceResult.count, points: mappedPoints)

            DispatchQueue.main.async {
                self.drawDots(result.points)
                self.onResult?(result)
            }
        }
    }

    private func finishProcessing(keepCapturedFrame: Bool) {
        stateQueue.async {
            self.captureState.finishProcessing(keepCapturedFrame: keepCapturedFrame)
            let isProcessing = self.captureState.isProcessing
            let isShowingCapturedFrame = self.captureState.isShowingCapturedFrame

            DispatchQueue.main.async {
                self.onProcessingChange?(isProcessing)
                self.onCaptureStateChange?(isShowingCapturedFrame)

                if !isShowingCapturedFrame {
                    self.frozenImageView.image = nil
                    self.frozenImageView.isHidden = true
                    self.dotLayer.path = nil
                }
            }
        }
    }

    private func drawDots(_ points: [CGPoint]) {
        let width = dotLayer.bounds.width
        let height = dotLayer.bounds.height
        guard width > 0, height > 0 else { return }

        let path = UIBezierPath()

        for point in points {
            let x = point.x * width
            let y = point.y * height

            let dot = UIBezierPath(
                arcCenter: CGPoint(x: x, y: y),
                radius: 4,
                startAngle: 0,
                endAngle: .pi * 2,
                clockwise: true
            )
            path.append(dot)
        }

        dotLayer.path = path.cgPath
    }

    private func setPreviewPortraitOrientation() {
        if let previewConnection = previewLayer.connection, previewConnection.isVideoOrientationSupported {
            previewConnection.videoOrientation = .portrait
        }
    }

    private func copyPixelBuffer(_ source: CVPixelBuffer) -> CVPixelBuffer? {
        let width = CVPixelBufferGetWidth(source)
        let height = CVPixelBufferGetHeight(source)
        let pixelFormat = CVPixelBufferGetPixelFormatType(source)

        var copied: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ]

        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            pixelFormat,
            attrs as CFDictionary,
            &copied
        )

        guard status == kCVReturnSuccess, let destination = copied else {
            return nil
        }

        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(destination, [])
        defer {
            CVPixelBufferUnlockBaseAddress(destination, [])
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
        }

        guard
            let sourceBase = CVPixelBufferGetBaseAddress(source),
            let destinationBase = CVPixelBufferGetBaseAddress(destination)
        else {
            return nil
        }

        let sourceBytesPerRow = CVPixelBufferGetBytesPerRow(source)
        let destinationBytesPerRow = CVPixelBufferGetBytesPerRow(destination)
        let copyBytesPerRow = min(sourceBytesPerRow, destinationBytesPerRow)

        let sourcePtr = sourceBase.assumingMemoryBound(to: UInt8.self)
        let destinationPtr = destinationBase.assumingMemoryBound(to: UInt8.self)

        for row in 0..<height {
            memcpy(
                destinationPtr.advanced(by: row * destinationBytesPerRow),
                sourcePtr.advanced(by: row * sourceBytesPerRow),
                copyBytesPerRow
            )
        }

        return destination
    }

    private func modelFrameRect(in bounds: CGRect) -> CGRect {
        let horizontalPadding: CGFloat = 16
        let topInset: CGFloat = 120
        let bottomReservedHeight: CGFloat = 280
        let targetSide: CGFloat = 352

        let availableWidth = max(1, bounds.width - (horizontalPadding * 2))
        let availableHeight = max(1, bounds.height - topInset - bottomReservedHeight)
        let side = min(targetSide, availableWidth, availableHeight)

        let x = (bounds.width - side) * 0.5
        let y = topInset + max(0, (availableHeight - side) * 0.5)
        return CGRect(x: x, y: y, width: side, height: side)
    }
}
