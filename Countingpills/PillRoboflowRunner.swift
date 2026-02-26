import CoreVideo
import Foundation
import Roboflow

private enum RoboflowConfigStore {
    static let apiKey = "roboflow.api.key"
    static let modelId = "roboflow.model.id"
    static let modelVersion = "roboflow.model.version"
}

struct RoboflowPillModelConfig {
    let apiKey: String
    let modelId: String
    let version: Int

    static let `default`: RoboflowPillModelConfig = {
        let defaults = UserDefaults.standard
        let env = ProcessInfo.processInfo.environment

        let apiKey =
            env["ROBOFLOW_API_KEY"] ??
            (Bundle.main.object(forInfoDictionaryKey: "ROBOFLOW_API_KEY") as? String) ??
            defaults.string(forKey: RoboflowConfigStore.apiKey) ??
            ""

        let modelId =
            env["ROBOFLOW_PILL_MODEL_ID"] ??
            (Bundle.main.object(forInfoDictionaryKey: "ROBOFLOW_PILL_MODEL_ID") as? String) ??
            defaults.string(forKey: RoboflowConfigStore.modelId) ??
            "pill_count-instance-segment"

        let versionString =
            env["ROBOFLOW_PILL_MODEL_VERSION"] ??
            (Bundle.main.object(forInfoDictionaryKey: "ROBOFLOW_PILL_MODEL_VERSION") as? String) ??
            defaults.string(forKey: RoboflowConfigStore.modelVersion) ??
            "8"
        let version = Int(versionString) ?? 8

        // Persist once so app relaunch (without Xcode scheme env) still works.
        if !apiKey.isEmpty {
            defaults.set(apiKey, forKey: RoboflowConfigStore.apiKey)
        }
        defaults.set(modelId, forKey: RoboflowConfigStore.modelId)
        defaults.set(String(version), forKey: RoboflowConfigStore.modelVersion)

        return RoboflowPillModelConfig(apiKey: apiKey, modelId: modelId, version: version)
    }()
}

final class PillRoboflowRunner {
    private let detectionThreshold: Double = 0.70
    private let config: RoboflowPillModelConfig
    private let sdk: RoboflowMobile
    private let queue = DispatchQueue(label: "pill.roboflow.runner.queue")
    private let loadStateQueue = DispatchQueue(label: "pill.roboflow.load.state.queue")

    private var loadedModel: RFModel?
    private var isLoadingModel = false
    private var lastLoadAttempt: Date = .distantPast

    init(config: RoboflowPillModelConfig = .default) {
        self.config = config
        self.sdk = RoboflowMobile(apiKey: config.apiKey)

        // Preload early so the first capture does not race model loading.
        queue.async { [weak self] in
            _ = self?.ensureModelLoaded()
        }
    }

    func detect(pixelBuffer: CVPixelBuffer) -> PillInferenceResult {
        guard ensureModelLoaded() else {
            #if DEBUG
            print("[PillDebug] roboflow-model-not-loaded")
            #endif
            return PillInferenceResult(count: 0, points: [], detections: [])
        }

        var finalDetections: [PillDetection] = []
        let semaphore = DispatchSemaphore(value: 0)

        queue.async { [weak self] in
            guard let self, let model = self.loadedModel else {
                semaphore.signal()
                return
            }

            model.detect(pixelBuffer: pixelBuffer) { detections, error in
                defer { semaphore.signal() }

                if let error {
                    #if DEBUG
                    print("[PillDebug] roboflow-detect-failed: \(error.localizedDescription)")
                    #endif
                    return
                }

                guard let rawDetections = detections else { return }
                finalDetections = self.parseDetections(rawDetections, pixelBuffer: pixelBuffer)
            }
        }

        _ = semaphore.wait(timeout: .now() + 6.0)

        return PillInferenceResult(
            count: finalDetections.count,
            points: finalDetections.map(\.point),
            detections: finalDetections
        )
    }

    func isModelReady() -> Bool {
        loadStateQueue.sync { loadedModel != nil }
    }

    private func ensureModelLoaded() -> Bool {
        if loadedModel != nil { return true }
        guard !config.apiKey.isEmpty else {
            #if DEBUG
            print("[PillDebug] roboflow-api-key-empty")
            #endif
            return false
        }

        // Allow retries when loading failed or timed out, but throttle spam.
        var shouldStartLoad = false
        loadStateQueue.sync {
            if self.loadedModel != nil {
                shouldStartLoad = false
                return
            }

            if self.isLoadingModel {
                shouldStartLoad = false
                return
            }

            let elapsed = Date().timeIntervalSince(self.lastLoadAttempt)
            if elapsed >= 1.0 {
                self.isLoadingModel = true
                self.lastLoadAttempt = Date()
                shouldStartLoad = true
            }
        }

        if !shouldStartLoad {
            return waitForModelLoad(timeout: 2.0)
        }

        let semaphore = DispatchSemaphore(value: 0)
        var success = false

        sdk.load(model: config.modelId, modelVersion: config.version) { [weak self] model, error, _, _ in
            defer { semaphore.signal() }

            guard let self else { return }

            if let error {
                #if DEBUG
                print("[PillDebug] roboflow-load-failed: \(error.localizedDescription)")
                #endif
                self.loadStateQueue.sync {
                    self.isLoadingModel = false
                }
                return
            }

            guard let model else { return }
            model.configure(
                threshold: self.detectionThreshold,
                overlap: 0.55,
                maxObjects: 220,
                processingMode: .balanced,
                maxNumberPoints: 120
            )

            self.loadStateQueue.sync {
                self.loadedModel = model
                self.isLoadingModel = false
            }
            success = true
            #if DEBUG
            print("[PillDebug] roboflow-load-ok model=\(self.config.modelId) version=\(self.config.version)")
            #endif
        }

        let completed = semaphore.wait(timeout: .now() + 10.0) == .success
        if !completed {
            loadStateQueue.sync {
                self.isLoadingModel = false
            }
            #if DEBUG
            print("[PillDebug] roboflow-load-timeout")
            #endif
            return false
        }

        if success { return true }
        return waitForModelLoad(timeout: 1.5)
    }

    private func waitForModelLoad(timeout: TimeInterval) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            let state = loadStateQueue.sync { (loadedModel != nil, isLoadingModel) }
            if state.0 { return true }
            if !state.1 { return false }
            usleep(60_000)
        }
        return loadStateQueue.sync { loadedModel != nil }
    }

    private func parseDetections(_ rawDetections: [RFPrediction], pixelBuffer: CVPixelBuffer) -> [PillDetection] {
        let width = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let height = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
        guard width > 0, height > 0 else { return [] }

        var parsed: [PillDetection] = []
        parsed.reserveCapacity(rawDetections.count)

        for detection in rawDetections {
            let values = detection.getValues()

            let confidence = (values["confidence"] as? Double) ?? 0
            if confidence < detectionThreshold { continue }

            let polygonCenter = polygonCenter(from: values["points"])

            let x = CGFloat((values["x"] as? Float) ?? 0)
            let y = CGFloat((values["y"] as? Float) ?? 0)
            let w = CGFloat((values["width"] as? Float) ?? 0)
            let h = CGFloat((values["height"] as? Float) ?? 0)

            guard w > 1, h > 1 else { continue }

            let centerX = polygonCenter?.x ?? (x + w * 0.5)
            let centerY = polygonCenter?.y ?? (y + h * 0.5)

            let normalizedX = min(1, max(0, centerX / width))
            let normalizedY = min(1, max(0, centerY / height))
            let meanSide = ((w + h) * 0.5)

            parsed.append(
                PillDetection(
                    point: CGPoint(x: normalizedX, y: normalizedY),
                    meanSide: meanSide,
                    avgScore: Float(confidence),
                    variantHits: 1
                )
            )
        }

        return deduplicate(parsed)
    }

    private func polygonCenter(from rawPoints: Any?) -> CGPoint? {
        guard let points = rawPoints as? [[String: Float]], !points.isEmpty else { return nil }

        var sumX: CGFloat = 0
        var sumY: CGFloat = 0
        var count: CGFloat = 0

        for point in points {
            guard let x = point["x"], let y = point["y"] else { continue }
            sumX += CGFloat(x)
            sumY += CGFloat(y)
            count += 1
        }

        guard count > 0 else { return nil }
        return CGPoint(x: sumX / count, y: sumY / count)
    }

    private func deduplicate(_ detections: [PillDetection]) -> [PillDetection] {
        guard detections.count > 1 else { return detections }

        let sorted = detections.sorted { lhs, rhs in
            if lhs.avgScore != rhs.avgScore { return lhs.avgScore > rhs.avgScore }
            return lhs.meanSide > rhs.meanSide
        }

        var kept: [PillDetection] = []
        kept.reserveCapacity(sorted.count)

        for candidate in sorted {
            let sideNorm = max(0.008, min(0.04, candidate.meanSide / 640.0 * 0.24))
            let radius2 = sideNorm * sideNorm

            var isDuplicate = false
            for existing in kept {
                let dx = candidate.point.x - existing.point.x
                let dy = candidate.point.y - existing.point.y
                if dx * dx + dy * dy <= radius2 {
                    isDuplicate = true
                    break
                }
            }

            if !isDuplicate {
                kept.append(candidate)
            }
        }

        return kept
    }
}
