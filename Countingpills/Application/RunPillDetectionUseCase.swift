import CoreVideo
import Foundation

protocol RunPillDetectionUseCase {
    func run(modelPixelBuffers: [CVPixelBuffer]) -> PillInferenceResult
}

final class DefaultRunPillDetectionUseCase: RunPillDetectionUseCase {
    private let postProcessor = PillDetectionPostProcessor()

    private lazy var runner: PillsOnnxRunner? = {
        do {
            let created = try PillsOnnxRunner()
            #if DEBUG
            print("[PillDebug] runner-init-ok")
            #endif
            return created
        } catch {
            #if DEBUG
            print("[PillDebug] runner-init-failed: \(error.localizedDescription)")
            #endif
            return nil
        }
    }()

    func run(modelPixelBuffers: [CVPixelBuffer]) -> PillInferenceResult {
        guard !modelPixelBuffers.isEmpty else {
            return PillInferenceResult(count: 0, points: [])
        }

        guard let runner else {
            #if DEBUG
            print("[PillDebug] runner-nil")
            #endif
            return PillInferenceResult(count: 0, points: [])
        }

        do {
            let output = try runner.run(pixelBuffers: modelPixelBuffers, scalingMode: .zeroToOne)
            #if DEBUG
            print("[PillDebug] runner-output(scale=0..1) shape=\(output.shape) values=\(output.values.count)")
            #endif
            var result = postProcessor.process(output, expectedVariants: modelPixelBuffers.count)

            if result.count == 0 {
                let fallbackOutput = try runner.run(pixelBuffers: modelPixelBuffers, scalingMode: .zeroTo255)
                #if DEBUG
                print("[PillDebug] runner-output(scale=0..255) shape=\(fallbackOutput.shape) values=\(fallbackOutput.values.count)")
                #endif
                let fallbackResult = postProcessor.process(fallbackOutput, expectedVariants: modelPixelBuffers.count)
                if fallbackResult.count > 0 {
                    result = fallbackResult
                    #if DEBUG
                    print("[PillDebug] selected-scale=0..255 count=\(fallbackResult.count)")
                    #endif
                }
            }

            return result
        } catch {
            #if DEBUG
            print("[PillDebug] runner-run-failed: \(error.localizedDescription)")
            #endif
            return PillInferenceResult(count: 0, points: [])
        }
    }
}
