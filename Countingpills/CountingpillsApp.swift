//
//  CountingpillsApp.swift
//  Countingpills
//
//  Created by 신찬솔 on 1/15/26.
//

import SwiftUI
import UIKit

final class OrientationLockedAppDelegate: NSObject, UIApplicationDelegate {
    func application(
        _ application: UIApplication,
        supportedInterfaceOrientationsFor window: UIWindow?
    ) -> UIInterfaceOrientationMask {
        .portrait
    }
}

@main
struct CountingpillsApp: App {
    @UIApplicationDelegateAdaptor(OrientationLockedAppDelegate.self)
    private var appDelegate

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
