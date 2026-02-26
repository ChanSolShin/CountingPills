import CoreVideo
import Foundation

protocol RunPillDetectionUseCase {
    func run(modelPixelBuffers: [CVPixelBuffer]) -> PillInferenceResult
}

final class DefaultRunPillDetectionUseCase: RunPillDetectionUseCase {
    private let roboflowRunner: PillRoboflowRunner

    init() {
        self.roboflowRunner = PillRoboflowRunner()
    }

    func run(modelPixelBuffers: [CVPixelBuffer]) -> PillInferenceResult {
        guard let firstBuffer = modelPixelBuffers.first else {
            return PillInferenceResult(count: 0, points: [], detections: [])
        }

        let result = roboflowRunner.detect(pixelBuffer: firstBuffer)
        #if DEBUG
        print("[PillDebug] roboflow-result count=\(result.count)")
        #endif
        return result
    }
}
