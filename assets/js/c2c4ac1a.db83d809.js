(window.webpackJsonp=window.webpackJsonp||[]).push([[15],{86:function(e,n,t){"use strict";t.r(n),t.d(n,"frontMatter",(function(){return o})),t.d(n,"metadata",(function(){return c})),t.d(n,"toc",(function(){return r})),t.d(n,"default",(function(){return b}));var l=t(3),a=t(7),i=(t(0),t(93)),o={id:"doc-setup-dev-env",title:"Setting Up Dev Environment"},c={unversionedId:"doc-setup-dev-env",id:"doc-setup-dev-env",isDocsHomePage:!1,title:"Setting Up Dev Environment",description:"Setting Up Dependencies",source:"@site/docs/doc-setup-dev-env.md",sourceDirName:".",slug:"/doc-setup-dev-env",permalink:"/OpenRace/doc-setup-dev-env",editUrl:"https://github.com/coderrect-inc/OpenRace/tree/develop/website/docs/doc-setup-dev-env.md",version:"current",frontMatter:{id:"doc-setup-dev-env",title:"Setting Up Dev Environment"},sidebar:"openraceSidebar",previous:{title:"Get Started with Coderrect OpenRace",permalink:"/OpenRace/"},next:{title:"Design Overview",permalink:"/OpenRace/doc-overview"}},r=[{value:"Setting Up Dependencies",id:"setting-up-dependencies",children:[{value:"Install Compiler",id:"install-compiler",children:[]},{value:"Install Conan",id:"install-conan",children:[]},{value:"Install LLVM 10.0.x",id:"install-llvm-100x",children:[]}]},{value:"Building OpenRace",id:"building-openrace",children:[]},{value:"Running Tests",id:"running-tests",children:[]},{value:"Using clang-format",id:"using-clang-format",children:[]}],s={toc:r};function b(e){var n=e.components,t=Object(a.a)(e,["components"]);return Object(i.b)("wrapper",Object(l.a)({},s,t,{components:n,mdxType:"MDXLayout"}),Object(i.b)("h2",{id:"setting-up-dependencies"},"Setting Up Dependencies"),Object(i.b)("p",null,"It is ",Object(i.b)("strong",{parentName:"p"},"highly")," recommended that development be done on Linux."),Object(i.b)("p",null,"We test all of our builds using Ubuntu 20.04 and most of our core team is using Manjaro, though any recent linux based OS should work."),Object(i.b)("p",null,"This guide will give instructions based on Ubuntu 20.04."),Object(i.b)("p",null,"To build follow this guide on setting up a dev environment you will need:"),Object(i.b)("ul",null,Object(i.b)("li",{parentName:"ul"},"A modern C++ compiler (gcc/clang)"),Object(i.b)("li",{parentName:"ul"},Object(i.b)("a",{parentName:"li",href:"https://ninja-build.org/"},"Ninja Build")),Object(i.b)("li",{parentName:"ul"},Object(i.b)("a",{parentName:"li",href:"https://conan.io/downloads.html"},"Conan")),Object(i.b)("li",{parentName:"ul"},"LLVM 10.0.X")),Object(i.b)("h3",{id:"install-compiler"},"Install Compiler"),Object(i.b)("p",null,"Most systems should already have gcc installed, but just in case, these commands can be used to install gcc. "),Object(i.b)("pre",null,Object(i.b)("code",{parentName:"pre",className:"language-shell"},"# Update packages \napt-get update\n# Install gcc and ninja\napt-get install -y build-essential ninja-build\n# Check that gcc is installed\ngcc --version\n")),Object(i.b)("h3",{id:"install-conan"},"Install Conan"),Object(i.b)("p",null,"Conan is used to automatically manage OpenRace's dependencies (except for LLVM)."),Object(i.b)("p",null,"If you already have python and pip installed, the easiest way to install conan is by running either of the following:"),Object(i.b)("pre",null,Object(i.b)("code",{parentName:"pre",className:"language-shell"},"# Python2\npip install conan\n# Python3\npip3 install conan\n")),Object(i.b)("p",null,"On Ubuntu, the binary can also be directly downloaded and installed."),Object(i.b)("pre",null,Object(i.b)("code",{parentName:"pre",className:"language-shell"},"# Download\ncurl -L -O https://github.com/conan-io/conan/releases/latest/download/conan-ubuntu-64.deb\n# Install\ndpkg -i conan-ubuntu-64.deb \n")),Object(i.b)("p",null,"For more information or examples on installing ",Object(i.b)("a",{parentName:"p",href:"https://conan.io/downloads.html"},"Conan"),", see their ",Object(i.b)("a",{parentName:"p",href:"https://docs.conan.io/en/latest/installation.html"},"installation instructions"),"."),Object(i.b)("h3",{id:"install-llvm-100x"},"Install LLVM 10.0.x"),Object(i.b)("p",null,"There are two ways to get LLVM:"),Object(i.b)("ul",null,Object(i.b)("li",{parentName:"ul"},"From package manager (Ubuntu 20.04 only)"),Object(i.b)("li",{parentName:"ul"},"Build from source")),Object(i.b)("p",null,"The easy/quick way is to install from package manager, though it may not work on all systems. In most cases LLVM will likely need to be built from source."),Object(i.b)("p",null,"In either case, LLVM will include a file named ",Object(i.b)("inlineCode",{parentName:"p"},"LLVMConfig.cmake"),". You will need to save the directory containing this file in order to build OpenRace."),Object(i.b)("p",null,"In this guide we save it into the ",Object(i.b)("inlineCode",{parentName:"p"},"LLVM_DIR")," environment variable."),Object(i.b)("h4",{id:"package-manager"},"Package Manager"),Object(i.b)("pre",null,Object(i.b)("code",{parentName:"pre",className:"language-shell"},"# Install LLVM 10\napt-get update\napt install -y llvm-10\n# Save location of LLVMConfig.cmake\nexport LLVM_DIR=/usr/lib/llvm-10/lib/cmake/llvm/\n")),Object(i.b)("h4",{id:"from-source"},"From Source"),Object(i.b)("pre",null,Object(i.b)("code",{parentName:"pre",className:"language-shell"},'# Get the source code\ngit clone --depth 1 -b llvmorg-10.0.1 https://github.com/llvm/llvm-project.git\nmkdir -p llvm-project/build && cd llvm-project/build\n# Configure the build with CMake\ncmake\n    -DLLVM_TARGETS_TO_BUILD="X86" \\\n    -DCMAKE_CXX_STANDARD="17" \\\n    -DLLVM_INCLUDE_EXAMPLES=OFF \\\n    -DLLVM_INCLUDE_TESTS=OFF \\\n    -DLLVM_INCLUDE_BENCHMARKS=OFF \\\n    -DLLVM_APPEND_VC_REV=OFF \\\n    -DLLVM_OPTIMIZED_TABLEGEN=ON \\\n    -DCMAKE_BUILD_TYPE=Release \\\n    -G Ninja \\\n    ../llvm\n# Build and Install\ncmake --build . --parallel\ncmake --build . --target install\n# Save location of LLVMConfig.cmake\nexport LLVM_DIR=$(pwd)/\n')),Object(i.b)("p",null,"There are a lot of CMake options to customize the LLVM build. See ",Object(i.b)("a",{parentName:"p",href:"https://www.llvm.org/docs/CMake.html"},"LLVM's page on CMake")," for more options."),Object(i.b)("p",null,"The important ones used above are:"),Object(i.b)("ul",null,Object(i.b)("li",{parentName:"ul"},Object(i.b)("inlineCode",{parentName:"li"},'LLVM_TARGETS_TO_BUILD="X86"'),Object(i.b)("br",{parentName:"li"}),"We only build for the X86 platform to save time"),Object(i.b)("li",{parentName:"ul"},Object(i.b)("inlineCode",{parentName:"li"},'CMAKE_CXX_STANDARD="17"'),Object(i.b)("br",{parentName:"li"}),"OpenRace is also using C++17"),Object(i.b)("li",{parentName:"ul"},Object(i.b)("inlineCode",{parentName:"li"},"CMAKE_BUILD_TYPE=Debug"),Object(i.b)("br",{parentName:"li"}),"Builds LLVM in Debug mode to make debugging easier"),Object(i.b)("li",{parentName:"ul"},Object(i.b)("inlineCode",{parentName:"li"},"-G Ninja"),Object(i.b)("br",{parentName:"li"}),"Building using Ninja Build")),Object(i.b)("p",null,"The rest are just some options set to save time/space when building."),Object(i.b)("h2",{id:"building-openrace"},"Building OpenRace"),Object(i.b)("p",null,"The recommended method of building the project for development is"),Object(i.b)("pre",null,Object(i.b)("code",{parentName:"pre",className:"language-shell"},"# Get the source code\ngit clone https://github.com/coderrect-inc/OpenRace.git\nmkdir build && cd build\n# Let conan build dependencies\nconan install ..\n# Configure build with cmake\ncmake \\\n    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \\\n    -DCMAKE_BUILD_TYPE=Debug \\\n    -DLLVM_DIR=$LLVM_DIR \\\n    -G Ninja \\\n    ..\n# Build OpenRace\ncmake --build . --parallel\n# Check that OpenRace was built\n./bin/openrace --help\n")),Object(i.b)("p",null,"The cmake options do the following:"),Object(i.b)("ul",null,Object(i.b)("li",{parentName:"ul"},Object(i.b)("inlineCode",{parentName:"li"},"CMAKE_EXPORT_COMPILE_COMMANDS=ON"),Object(i.b)("br",{parentName:"li"}),"produces a ",Object(i.b)("inlineCode",{parentName:"li"},"compile_commands.json")," file in the build directory. Most IDEs can be set up to use this file for neat IDE features."),Object(i.b)("li",{parentName:"ul"},Object(i.b)("inlineCode",{parentName:"li"},"CMAKE_BUILD_TYPE=Debug"),Object(i.b)("br",{parentName:"li"}),"Builds the project in debug mode. This makes it is easier to debug if/when issues occur."),Object(i.b)("li",{parentName:"ul"},Object(i.b)("inlineCode",{parentName:"li"},"LLVM_DIR=$LLVM_DIR"),Object(i.b)("br",{parentName:"li"}),"Should point to a directory containing ",Object(i.b)("inlineCode",{parentName:"li"},"LLVMConfig.cmake"),'. See the "Install LLVM 10.0.X" section above.')),Object(i.b)("h2",{id:"running-tests"},"Running Tests"),Object(i.b)("p",null,"From the build directory, run ",Object(i.b)("inlineCode",{parentName:"p"},"ctest"),". This should automatically handle running tests from the correct directory. "),Object(i.b)("pre",null,Object(i.b)("code",{parentName:"pre",className:"language-shell"},"OpenRace/build> ctest\n...\n100% tests passed, 0 tests failed out of 25\nTotal Test time (real) =  29.24 sec\n")),Object(i.b)("p",null,"Tests can also be executed manually using the",Object(i.b)("inlineCode",{parentName:"p"},"bin/tester")," executable. "),Object(i.b)("p",null,"Many of the tests read IR files and expect to be run from the ",Object(i.b)("inlineCode",{parentName:"p"},"/tests/data")," directory. Keep this in mind if running tests manually. "),Object(i.b)("pre",null,Object(i.b)("code",{parentName:"pre",className:"language-shell"},"OpenRace/tests/data> ../../build/bin/tester\n...\n100% tests passed, 0 tests failed out of 25\nTotal Test time (real) =  29.24 sec\n")),Object(i.b)("p",null,"We use ",Object(i.b)("a",{parentName:"p",href:"https://github.com/catchorg/Catch2"},"Catch2")," for testing. See their documentation for more options on running their tests."),Object(i.b)("p",null,"It is expected that all tests will always pass in the main branch of the project."),Object(i.b)("h2",{id:"using-clang-format"},"Using clang-format"),Object(i.b)("p",null,"All code should be formatted according to the ",Object(i.b)("inlineCode",{parentName:"p"},".clang-format")," file at the project root."),Object(i.b)("p",null,"Coderrect OpenRace adopts ",Object(i.b)("a",{parentName:"p",href:"https://google.github.io/styleguide/cppguide.html"},"Google's C++ Style Guide")," with some small customizations."),Object(i.b)("p",null,"Most IDEs can be set to run clang-format automatically. Check the settings for your IDE on how to set this up."),Object(i.b)("p",null,"Worst case, clang format can be run manually on an individual file"),Object(i.b)("pre",null,Object(i.b)("code",{parentName:"pre"},"clang-format -i -style=file file.cpp\n")),Object(i.b)("p",null,"Or on the entire project directory (careful to run this from within this project's directory as it will recursively overwrite all files ending in .h or .cpp in this directory and all subdirectories)."),Object(i.b)("pre",null,Object(i.b)("code",{parentName:"pre"},"cd OpenRace/\nfind . -iname *.h -o -iname *.cpp | xargs clang-format -i -style=file\n")))}b.isMDXComponent=!0}}]);