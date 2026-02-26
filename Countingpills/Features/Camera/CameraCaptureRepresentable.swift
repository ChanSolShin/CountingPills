import SwiftUI

struct CameraCaptureRepresentable: UIViewControllerRepresentable {
    let captureToken: Int
    let retakeToken: Int
    @Binding var pillCount: Int
    @Binding var isProcessing: Bool
    @Binding var isShowingCapturedFrame: Bool
    @Binding var isModelReady: Bool

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    func makeUIViewController(context: Context) -> CameraCaptureViewController {
        let viewController = CameraCaptureViewController()
        viewController.onResult = { result in
            DispatchQueue.main.async {
                self.pillCount = result.count
            }
        }
        viewController.onProcessingChange = { processing in
            DispatchQueue.main.async {
                self.isProcessing = processing
            }
        }
        viewController.onCaptureStateChange = { showingCaptured in
            DispatchQueue.main.async {
                self.isShowingCapturedFrame = showingCaptured
            }
        }
        viewController.onModelReadyChange = { isReady in
            DispatchQueue.main.async {
                self.isModelReady = isReady
            }
        }
        return viewController
    }

    func updateUIViewController(_ uiViewController: CameraCaptureViewController, context: Context) {
        if context.coordinator.lastCaptureToken != captureToken {
            context.coordinator.lastCaptureToken = captureToken
            uiViewController.captureCurrentFrame()
        }

        if context.coordinator.lastRetakeToken != retakeToken {
            context.coordinator.lastRetakeToken = retakeToken
            uiViewController.resumeCameraForRetake()
        }
    }

    static func dismantleUIViewController(_ uiViewController: CameraCaptureViewController, coordinator: Coordinator) {
        uiViewController.stopSession()
    }

    final class Coordinator {
        var lastCaptureToken: Int = 0
        var lastRetakeToken: Int = 0
    }
}
