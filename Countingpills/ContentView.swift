import SwiftUI

struct ContentView: View {
    @AppStorage("hasSeenOnboarding") private var hasSeenOnboarding = false
    @State private var isShowingOnboarding = false

    var body: some View {
        CameraCaptureScreen()
            .ignoresSafeArea()
            .fullScreenCover(isPresented: $isShowingOnboarding) {
                OnboardingView {
                    hasSeenOnboarding = true
                    isShowingOnboarding = false
                }
            }
            .onAppear {
                if !hasSeenOnboarding {
                    isShowingOnboarding = true
                }
            }
    }
}
