import CoreImage
import CoreVideo
import Foundation
import OnnxRuntimeBindings

struct ModelOutputTensor {
    let values: [Float]
    let shape: [Int]
}

final class PillsOnnxRunner {
    let inputWidth = 640
    let inputHeight = 640
    let modelBatchSize = 8

    private let env: ORTEnv
    private let session: ORTSession
    private let ciContext = CIContext()

    private let inputName: String
    private let outputName: String

    private let perImageElements: Int
    private var inputFloatBuffer: [Float]
    private var resizedPixelBuffer: CVPixelBuffer

    init() throws {
        env = try ORTEnv(loggingLevel: ORTLoggingLevel.warning)

        guard let modelPath = Bundle.main.path(forResource: "best_model", ofType: "onnx") else {
            throw NSError(
                domain: "Model",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "best_model.onnx not found in bundle"]
            )
        }

        let options = try ORTSessionOptions()
        session = try ORTSession(env: env, modelPath: modelPath, sessionOptions: options)
        inputName = (try? session.inputNames().first) ?? "images"
        outputName = (try? session.outputNames().first) ?? "output0"

        perImageElements = 3 * inputWidth * inputHeight
        inputFloatBuffer = [Float](repeating: 0, count: modelBatchSize * perImageElements)

        guard let buffer = Self.makePixelBuffer(width: inputWidth, height: inputHeight) else {
            throw NSError(
                domain: "Preprocess",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Failed to allocate resized pixel buffer"]
            )
        }
        resizedPixelBuffer = buffer
    }

    enum InputScalingMode {
        case zeroToOne
        case zeroTo255
    }

    func run(pixelBuffers: [CVPixelBuffer], scalingMode: InputScalingMode = .zeroToOne) throws -> ModelOutputTensor {
        guard !pixelBuffers.isEmpty else {
            throw NSError(
                domain: "Input",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "At least one pixel buffer is required"]
            )
        }

        for batchIndex in 0..<modelBatchSize {
            let source = pixelBuffers[min(batchIndex, pixelBuffers.count - 1)]
            try preprocess(pixelBuffer: source, scalingMode: scalingMode, batchIndex: batchIndex)
        }

        let inputShape: [NSNumber] = [
            NSNumber(value: modelBatchSize),
            3,
            NSNumber(value: inputHeight),
            NSNumber(value: inputWidth)
        ]

        let inputData = inputFloatBuffer.withUnsafeBytes {
            NSMutableData(bytes: $0.baseAddress, length: $0.count)
        }

        let inputTensor = try ORTValue(
            tensorData: inputData,
            elementType: ORTTensorElementDataType.float,
            shape: inputShape
        )

        let outputs = try session.run(
            withInputs: [inputName: inputTensor],
            outputNames: [outputName],
            runOptions: nil
        )

        guard let outputTensor = outputs[outputName] else {
            throw NSError(domain: "Output", code: -1, userInfo: [NSLocalizedDescriptionKey: "Missing model output tensor"])
        }

        let data = try outputTensor.tensorData() as Data
        let values = data.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: Float.self))
        }

        let shapeInfo = try? outputTensor.tensorTypeAndShapeInfo()
        let outputShape = shapeInfo?.shape.map { $0.intValue } ?? []

        return ModelOutputTensor(values: values, shape: outputShape)
    }

    private func preprocess(
        pixelBuffer: CVPixelBuffer,
        scalingMode: InputScalingMode,
        batchIndex: Int
    ) throws {
        let image = CIImage(cvPixelBuffer: pixelBuffer)
        let extent = image.extent
        guard extent.width > 0, extent.height > 0 else {
            throw NSError(
                domain: "Preprocess",
                code: -2,
                userInfo: [NSLocalizedDescriptionKey: "Invalid input image extent"]
            )
        }

        let scaleX = CGFloat(inputWidth) / extent.width
        let scaleY = CGFloat(inputHeight) / extent.height
        let resized = image.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
        ciContext.render(resized, to: resizedPixelBuffer)

        CVPixelBufferLockBaseAddress(resizedPixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(resizedPixelBuffer, .readOnly) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(resizedPixelBuffer) else {
            throw NSError(
                domain: "Preprocess",
                code: -3,
                userInfo: [NSLocalizedDescriptionKey: "No base address in resized pixel buffer"]
            )
        }

        let bytesPerRow = CVPixelBufferGetBytesPerRow(resizedPixelBuffer)
        let raw = baseAddress.assumingMemoryBound(to: UInt8.self)

        let planePixels = inputWidth * inputHeight
        let batchOffset = batchIndex * perImageElements
        let rOffset = batchOffset
        let gOffset = batchOffset + planePixels
        let bOffset = batchOffset + 2 * planePixels

        let scale: Float = (scalingMode == .zeroToOne) ? (1.0 / 255.0) : 1.0

        for y in 0..<inputHeight {
            let row = raw.advanced(by: y * bytesPerRow)
            let rowOffset = y * inputWidth

            for x in 0..<inputWidth {
                let pixelOffset = x * 4
                let idx = rowOffset + x

                let b = row[pixelOffset]
                let g = row[pixelOffset + 1]
                let r = row[pixelOffset + 2]

                inputFloatBuffer[rOffset + idx] = Float(r) * scale
                inputFloatBuffer[gOffset + idx] = Float(g) * scale
                inputFloatBuffer[bOffset + idx] = Float(b) * scale
            }
        }
    }

    private static func makePixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
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
}
