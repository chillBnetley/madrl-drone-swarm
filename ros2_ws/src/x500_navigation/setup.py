from setuptools import setup

package_name = "x500_navigation"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="Moemen Hezzi",
    maintainer_email="momenhzez46@gmail.com",
    description="Autonomous navigation and PX4 offboard control for the X500 quadrotor",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "square_no_odom = x500_navigation.square_no_odom:main",
            "offboard_demo = x500_navigation.offboard_demo:main",
        ],
    },
)
