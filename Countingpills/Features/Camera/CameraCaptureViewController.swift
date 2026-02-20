import AVFoundation
import CoreImage
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

    private let ciContext = CIContext()
    private let postProcessor = PillDetectionPostProcessor()
    private let framePreprocessor = PillFramePreprocessor(modelSide: 640)

    private let previewContainerView = UIView()
    private let frameBorderLayer = CAShapeLayer()
    private let dotLayer = CAShapeLayer()
    private let frozenImageView = UIImageView()

    private var isProcessing = false
    private var isSessionConfigured = false
    private let modelRenderSize = CGSize(width: 640, height: 640)

    private var latestPixelBuffer: CVPixelBuffer?

    private lazy var runner: PillsOnnxRunner? = {
        do {
            let r = try PillsOnnxRunner()
            #if DEBUG
            print("[PillDebug] runner-init-ok")
            #endif
            return r
        } catch {
            #if DEBUG
            print("[PillDebug] runner-init-failed: \(error.localizedDescription)")
            #endif
            return nil
        }
    }()

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
            guard !self.isProcessing else { return }
            self.isProcessing = true
            guard
                let sourcePixelBuffer = self.latestBufferQueue.sync(execute: { self.latestPixelBuffer }),
                let ownedPixelBuffer = self.copyPixelBuffer(sourcePixelBuffer)
            else {
                #if DEBUG
                print("[PillDebug] capture-failed: latestPixelBuffer unavailable")
                #endif
                self.finishProcessing(resetCaptureState: false)
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
            self.isProcessing = false
        }

        DispatchQueue.main.async {
            self.frozenImageView.image = nil
            self.frozenImageView.isHidden = true
            self.dotLayer.path = nil
            self.onCaptureStateChange?(false)
            self.onProcessingChange?(false)
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
        guard let prepared = autoreleasepool(invoking: { self.prepareCapturedFrame(from: ownedPixelBuffer) }) else {
            finishProcessing(resetCaptureState: false)
            return
        }

        DispatchQueue.main.async {
            self.frozenImageView.image = prepared.displayImage
            self.frozenImageView.isHidden = false
            self.onCaptureStateChange?(true)
        }

        autoreleasepool {
            defer {
                self.finishProcessing(resetCaptureState: true)
            }

            guard let runner = self.runner else {
                #if DEBUG
                print("[PillDebug] runner-nil")
                #endif
                DispatchQueue.main.async {
                    self.dotLayer.path = nil
                    self.onResult?(PillInferenceResult(count: 0, points: []))
                }
                return
            }

            do {
                let output = try runner.run(pixelBuffers: prepared.modelPixelBuffers, scalingMode: .zeroToOne)
                #if DEBUG
                print("[PillDebug] runner-output(scale=0..1) shape=\(output.shape) values=\(output.values.count)")
                #endif
                var result = self.postProcessor.process(output, expectedVariants: prepared.modelPixelBuffers.count)

                if result.count == 0 {
                    let fallbackOutput = try runner.run(pixelBuffers: prepared.modelPixelBuffers, scalingMode: .zeroTo255)
                    #if DEBUG
                    print("[PillDebug] runner-output(scale=0..255) shape=\(fallbackOutput.shape) values=\(fallbackOutput.values.count)")
                    #endif
                    let fallbackResult = self.postProcessor.process(
                        fallbackOutput,
                        expectedVariants: prepared.modelPixelBuffers.count
                    )
                    if fallbackResult.count > 0 {
                        result = fallbackResult
                        #if DEBUG
                        print("[PillDebug] selected-scale=0..255 count=\(fallbackResult.count)")
                        #endif
                    }
                }

                let mappedPoints = self.mapPointsToDisplay(
                    result.points,
                    roiRect: prepared.inferenceROI,
                    edgeTrimRatio: prepared.edgeTrimRatio
                )
                result = PillInferenceResult(count: result.count, points: mappedPoints)

                DispatchQueue.main.async {
                    self.drawDots(result.points)
                    self.onResult?(result)
                }
            } catch {
                #if DEBUG
                print("[PillDebug] runner-run-failed: \(error.localizedDescription)")
                #endif
                DispatchQueue.main.async {
                    self.dotLayer.path = nil
                    self.onResult?(PillInferenceResult(count: 0, points: []))
                }
            }
        }
    }

    private func finishProcessing(resetCaptureState: Bool) {
        stateQueue.async {
            self.isProcessing = false
            DispatchQueue.main.async {
                self.onProcessingChange?(false)
                if resetCaptureState == false {
                    self.frozenImageView.image = nil
                    self.frozenImageView.isHidden = true
                    self.onCaptureStateChange?(false)
                }
            }
        }
    }

    private func prepareCapturedFrame(from pixelBuffer: CVPixelBuffer) -> PreparedSquareFrame? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let extent = ciImage.extent.integral
        let side = min(extent.width, extent.height)
        let cropRect = CGRect(
            x: extent.midX - side * 0.5,
            y: extent.midY - side * 0.5,
            width: side,
            height: side
        ).intersection(extent).integral

        guard !cropRect.isNull, cropRect.width > 0, cropRect.height > 0 else {
            return nil
        }

        let cropped = ciImage.cropped(to: cropRect)
        let translated = cropped.transformed(
            by: CGAffineTransform(
                translationX: -cropped.extent.origin.x,
                y: -cropped.extent.origin.y
            )
        )

        let sourceExtent = translated.extent.integral
        guard sourceExtent.width > 0, sourceExtent.height > 0 else { return nil }

        let modelScaleX = modelRenderSize.width / sourceExtent.width
        let modelScaleY = modelRenderSize.height / sourceExtent.height
        let resizedSquare = translated.transformed(by: CGAffineTransform(scaleX: modelScaleX, y: modelScaleY))
        let preprocessed = framePreprocessor.prepareForInference(from: resizedSquare, ciContext: ciContext)

        var modelPixelBuffers: [CVPixelBuffer] = []
        modelPixelBuffers.reserveCapacity(preprocessed.variantImages.count)
        for variantImage in preprocessed.variantImages {
            guard let buffer = makePixelBuffer(
                width: Int(modelRenderSize.width),
                height: Int(modelRenderSize.height)
            ) else {
                return nil
            }
            ciContext.render(variantImage, to: buffer)
            modelPixelBuffers.append(buffer)
        }
        guard !modelPixelBuffers.isEmpty else { return nil }

        #if DEBUG
        let confidenceText: String
        if let confidence = preprocessed.roiConfidence {
            confidenceText = String(format: "%.2f", confidence)
        } else {
            confidenceText = "-"
        }
        if let roi = preprocessed.roiRect {
            print(
                "[PillDebug] preprocess-roi[\(preprocessed.roiSource)] x=\(Int(roi.origin.x)) y=\(Int(roi.origin.y)) " +
                "w=\(Int(roi.width)) h=\(Int(roi.height)) conf=\(confidenceText) variants=\(modelPixelBuffers.count)"
            )
        } else {
            print("[PillDebug] preprocess-roi[\(preprocessed.roiSource)] none conf=\(confidenceText) variants=\(modelPixelBuffers.count)")
        }
        #endif

        let displayRect = CGRect(origin: .zero, size: modelRenderSize)
        guard let displayCGImage = ciContext.createCGImage(resizedSquare, from: displayRect) else {
            return nil
        }

        return PreparedSquareFrame(
            displayImage: UIImage(cgImage: displayCGImage),
            modelPixelBuffers: modelPixelBuffers,
            inferenceROI: preprocessed.roiRect ?? displayRect,
            edgeTrimRatio: preprocessed.edgeTrimRatio
        )
    }

    private func mapPointsToDisplay(
        _ points: [CGPoint],
        roiRect: CGRect,
        edgeTrimRatio: CGFloat
    ) -> [CGPoint] {
        let modelSide = modelRenderSize.width
        guard modelSide > 0 else { return points }

        let trim = max(0, min(modelSide * 0.2, modelSide * edgeTrimRatio))
        let trimmedSide = max(1, modelSide - trim * 2)
        let trimScale = trimmedSide / modelSide

        return points.map { point in
            let xModel = CGFloat(point.x) * modelSide
            let yModel = CGFloat(point.y) * modelSide

            let xRoi = xModel * trimScale + trim
            let yRoi = yModel * trimScale + trim

            let xDisplay = roiRect.minX + (xRoi / modelSide) * roiRect.width
            let yDisplay = roiRect.minY + (yRoi / modelSide) * roiRect.height

            let nx = max(0, min(1, xDisplay / modelSide))
            let ny = max(0, min(1, yDisplay / modelSide))
            return CGPoint(x: nx, y: ny)
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

    private func makePixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
        var pixelBuffer: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ]

        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            attrs as CFDictionary,
            &pixelBuffer
        )

        guard status == kCVReturnSuccess else { return nil }
        return pixelBuffer
    }

    private func modelFrameRect(in bounds: CGRect) -> CGRect {
        let availableWidth = bounds.width - 32
        let availableHeight = bounds.height - 280
        let side = max(220, min(availableWidth, availableHeight))

        let x = (bounds.width - side) * 0.5
        let y = max(120, (bounds.height - side) * 0.5)
        return CGRect(x: x, y: y, width: side, height: side)
    }
}

private struct PreparedSquareFrame {
    let displayImage: UIImage
    let modelPixelBuffers: [CVPixelBuffer]
    let inferenceROI: CGRect
    let edgeTrimRatio: CGFloat
}
