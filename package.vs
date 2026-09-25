// The 'tensor' repository: the n-d array, on a device. See README.md.
import PackageDescription

let package = Package(
    name: "tensor",
    platforms: [
        .macOS(.v13),
    ],
    products: [
        .library(name: "tensor", targets: ["tensor"]),
        .executable(name: "test-tensor", targets: ["test_tensor"]),
    ],
    targets: [
        .target(
            name: "tensor",
            path: "tensor"
        ),
        .executableTarget(
            name: "test_tensor",
            dependencies: ["tensor"],
            path: "tests/tensor"
        ),
    ]
)
