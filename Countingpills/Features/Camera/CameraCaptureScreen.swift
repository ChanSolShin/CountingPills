import SwiftUI

struct CameraCaptureScreen: View {
    @State private var captureToken = 0
    @State private var retakeToken = 0
    @State private var pillCount = 0
    @State private var isProcessing = false
    @State private var isShowingCapturedFrame = false

    var body: some View {
        ZStack {
            CameraCaptureRepresentable(
                captureToken: captureToken,
                retakeToken: retakeToken,
                pillCount: $pillCount,
                isProcessing: $isProcessing,
                isShowingCapturedFrame: $isShowingCapturedFrame
            )
            .ignoresSafeArea()

            VStack {
                Spacer()

                VStack(spacing: 25) {
                    if isShowingCapturedFrame && !isProcessing {
                        Text("\(pillCount)")
                            .font(.system(size: 36, weight: .bold))
                            .foregroundColor(.green)
                            .shadow(color: .black.opacity(0.75), radius: 4, x: 0, y: 2)
                    }

                    if isShowingCapturedFrame && !isProcessing {
                        Button(action: {
                            pillCount = 0
                            retakeToken &+= 1
                        }) {
                            ZStack {
                                Circle()
                                    .fill(.white)
                                    .frame(width: 78, height: 78)

                                Circle()
                                    .stroke(.black.opacity(0.8), lineWidth: 2)
                                    .frame(width: 66, height: 66)

                                Image(systemName: "arrow.counterclockwise")
                                    .font(.system(size: 28, weight: .bold))
                                    .foregroundColor(.black)
                            }
                        }
                        .disabled(isProcessing)
                        .opacity(isProcessing ? 0.6 : 1.0)
                    } else {
                        Button(action: { captureToken &+= 1 }) {
                            ZStack {
                                Circle()
                                    .fill(.white)
                                    .frame(width: 78, height: 78)

                                Circle()
                                    .stroke(.black.opacity(0.8), lineWidth: 2)
                                    .frame(width: 66, height: 66)
                            }
                        }
                        .disabled(isProcessing || isShowingCapturedFrame)
                        .opacity((isProcessing || isShowingCapturedFrame) ? 0.6 : 1.0)
                    }
                }
                .padding(.bottom, 36)
            }

            if isProcessing {
                VStack(spacing: 8) {
                    ProgressView()
                }
                .padding(12)
                .background(Color.black.opacity(0.7))
                .cornerRadius(12)
            }
        }
    }
}
