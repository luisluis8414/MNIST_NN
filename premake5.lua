workspace "MultiLayerPerception"
   architecture "x64"
   configurations { "Debug", "Release" }

-- MLP DLL project
project "mlp"
   kind "SharedLib"        -- Build as a DLL
   language "C++"
   cppdialect "C++17"
   staticruntime "off"     -- Use dynamic runtime libraries

   -- Output directories for binaries and intermediate object files
   targetdir "bin/%{cfg.platform}/%{cfg.buildcfg}/mlp"
   objdir "obj/%{cfg.platform}/%{cfg.buildcfg}/mlp"

   -- Specify the files to include in the build
   files { "mlp/include/**.h", "mlp/src/**.cpp" }

   -- Include directories for header files
   includedirs { "mlp/include" }

   -- Platform-specific settings
   filter "system:windows"
      systemversion "latest"  -- Use the latest Windows SDK
      defines { "PLATFORM_WINDOWS", "MLP_EXPORT" }  -- Define macros for Windows and DLL export

   -- Debug configuration settings
   filter "configurations:Debug"
      defines "DEBUG"         -- Define the DEBUG macro
      runtime "Debug"         -- Use the debug runtime library
      symbols "on"            -- Enable debug symbols for debugging

   -- Release configuration settings
   filter "configurations:Release"
      defines "NDEBUG"        -- Define the NDEBUG macro (no debug)
      runtime "Release"       -- Use the release runtime library
      optimize "on"           -- Enable optimizations for better performance

-- Client / Trainer app project
project "MNIST"
   kind "ConsoleApp"         -- Build as a console application
   language "C++"
   cppdialect "C++17"
   staticruntime "off"       -- Use dynamic runtime libraries

   -- Output directories for binaries and intermediate object files
   targetdir "bin/%{cfg.platform}/%{cfg.buildcfg}/MNIST"
   objdir "obj/%{cfg.platform}/%{cfg.buildcfg}/MNIST"

   -- Specify the files to include in the build
   files { "MNIST/src/**.hpp", "MNIST/src/**.cpp" }

   -- Include directories for header files
   includedirs { "mlp/include", "MNIST/src", "include" }

   -- Link against the MLP DLL and OpenCV
   libdirs { "bin/%{cfg.platform}/%{cfg.buildcfg}/mlp", "lib" }
   links { "mlp", "opencv_world4110d" }

   debugdir "%{cfg.targetdir}"
   
   postbuildcommands {
      "{COPY} bin/%{cfg.platform}/%{cfg.buildcfg}/mlp/mlp.dll %{cfg.targetdir}",
      "{COPY} MNIST/resources %{cfg.targetdir}/resources",
      "{COPY} MNIST/models %{cfg.targetdir}/models",
   }

   -- Platform-specific settings
   filter "system:windows"
      systemversion "latest"  -- Use the latest Windows SDK
      defines { "PLATFORM_WINDOWS" }

   -- Debug configuration settings
   filter "configurations:Debug"
      defines "DEBUG"         -- Define the DEBUG macro
      runtime "Debug"         -- Use the debug runtime library
      symbols "on"            -- Enable debug symbols for debugging
      postbuildcommands {
         "{COPY} vendor/opencv/build/x64/vc16/bin/opencv_world4110d.dll %{cfg.targetdir}",
      }

   -- Release configuration settings
   filter "configurations:Release"
      defines "NDEBUG"        -- Define the NDEBUG macro (no debug)
      runtime "Release"       -- Use the release runtime library
      optimize "on"           -- Enable optimizations for better performance
      postbuildcommands {
         "{COPY} vendor/opencv/build/x64/vc16/bin/opencv_world4110.dll %{cfg.targetdir}",
      }
