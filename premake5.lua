workspace "MultiLayerPerception"
   architecture "x64"
   configurations { "Debug", "Release" }

project "mlp"
   kind "SharedLib"
   language "C++"
   cppdialect "C++17"
   staticruntime "off"
   targetdir "bin/%{cfg.platform}/%{cfg.buildcfg}/mlp"
   objdir "obj/%{cfg.platform}/%{cfg.buildcfg}/mlp"
   files { "mlp/include/**.h", "mlp/src/**.cpp" }
   includedirs { "mlp/include" }
   filter "system:windows"
      systemversion "latest"
      defines { "PLATFORM_WINDOWS", "MLP_EXPORT" }
   filter "configurations:Debug"
      defines "DEBUG"
      runtime "Debug"
      symbols "on"
   filter "configurations:Release"
      defines "NDEBUG"
      runtime "Release"
      optimize "on"

project "MNIST"
   kind "ConsoleApp"
   language "C++"
   cppdialect "C++17"
   staticruntime "off"
   targetdir "bin/%{cfg.platform}/%{cfg.buildcfg}/MNIST"
   objdir "obj/%{cfg.platform}/%{cfg.buildcfg}/MNIST"
   files { "MNIST/src/**.hpp", "MNIST/src/**.cpp" }
   includedirs { "mlp/include", "MNIST/src", "include" }
   libdirs { "bin/%{cfg.platform}/%{cfg.buildcfg}/mlp", "lib" }
   links { "mlp", "opencv_world4110d" }
   debugdir "%{cfg.targetdir}"
   postbuildcommands {
      "{COPY} bin/%{cfg.platform}/%{cfg.buildcfg}/mlp/mlp.dll %{cfg.targetdir}",
      "{COPY} MNIST/resources %{cfg.targetdir}/resources",
      "{COPY} MNIST/models %{cfg.targetdir}/models",
   }
   filter "system:windows"
      systemversion "latest"
      defines { "PLATFORM_WINDOWS" }
   filter "configurations:Debug"
      defines "DEBUG"
      runtime "Debug"
      symbols "on"
      postbuildcommands {
         "{COPY} vendor/opencv/build/x64/vc16/bin/opencv_world4110d.dll %{cfg.targetdir}",
      }
   filter "configurations:Release"
      defines "NDEBUG"
      runtime "Release"
      optimize "on"
      postbuildcommands {
         "{COPY} vendor/opencv/build/x64/vc16/bin/opencv_world4110.dll %{cfg.targetdir}",
      }

project "draw_and_predict"
   kind "ConsoleApp"
   language "C++"
   cppdialect "C++17"
   staticruntime "off"
   targetdir "bin/%{cfg.platform}/%{cfg.buildcfg}/draw_and_predict"
   objdir "obj/%{cfg.platform}/%{cfg.buildcfg}/draw_and_predict"
   files { "draw_and_predict/src/**.hpp", "draw_and_predict/src/**.cpp" }
   includedirs { "mlp/include", "draw_and_predict/src", "include" }
   libdirs { "bin/%{cfg.platform}/%{cfg.buildcfg}/mlp", "lib" }
   links { "mlp", "opencv_world4110d" }
   debugdir "%{cfg.targetdir}"
   postbuildcommands {
      "{COPY} bin/%{cfg.platform}/%{cfg.buildcfg}/mlp/mlp.dll %{cfg.targetdir}",
      "{COPY} MNIST/models %{cfg.targetdir}/models",
   }
   filter "system:windows"
      systemversion "latest"
      defines { "PLATFORM_WINDOWS" }
   filter "configurations:Debug"
      defines "DEBUG"
      runtime "Debug"
      symbols "on"
      postbuildcommands {
         "{COPY} vendor/opencv/build/x64/vc16/bin/opencv_world4110d.dll %{cfg.targetdir}",
      }
   filter "configurations:Release"
      defines "NDEBUG"
      runtime "Release"
      optimize "on"
      postbuildcommands {
         "{COPY} vendor/opencv/build/x64/vc16/bin/opencv_world4110.dll %{cfg.targetdir}",
      }

project "booster_landing"
   kind "ConsoleApp"
   language "C++"
   cppdialect "C++17"
   staticruntime "off"
   targetdir "bin/%{cfg.platform}/%{cfg.buildcfg}/booster_landing"
   objdir "obj/%{cfg.platform}/%{cfg.buildcfg}/booster_landing"
   files { "booster_landing/src/**.hpp", "booster_landing/src/**.cpp" }
   includedirs { "mlp/include", "booster_landing/src", "include" }
   libdirs { "bin/%{cfg.platform}/%{cfg.buildcfg}/mlp", "lib" }
   links { "mlp", "opencv_world4110d" }
   debugdir "%{cfg.targetdir}"
   postbuildcommands {
      "{COPY} bin/%{cfg.platform}/%{cfg.buildcfg}/mlp/mlp.dll %{cfg.targetdir}",
      "{COPY} booster_landing/resources %{cfg.targetdir}/resources",
   }
   filter "system:windows"
      systemversion "latest"
      defines { "PLATFORM_WINDOWS" }
   filter "configurations:Debug"
      defines "DEBUG"
      runtime "Debug"
      symbols "on"
      postbuildcommands {
         "{COPY} vendor/SFML-3.0.0/bin/sfml-graphics-d-3.dll %{cfg.targetdir}",
         "{COPY} vendor/SFML-3.0.0/bin/sfml-window-d-3.dll %{cfg.targetdir}",
         "{COPY} vendor/SFML-3.0.0/bin/sfml-system-d-3.dll %{cfg.targetdir}",
     }
      links {
         "sfml-graphics-d",
         "sfml-window-d",
         "sfml-system-d",
      }
   filter "configurations:Release"
      defines "NDEBUG"
      runtime "Release"
      optimize "on"
      postbuildcommands {
         "{COPY} vendor/SFML-3.0.0/bin/sfml-graphics-3.dll %{cfg.targetdir}",
         "{COPY} vendor/SFML-3.0.0/bin/sfml-window-3.dll %{cfg.targetdir}",
         "{COPY} vendor/SFML-3.0.0/bin/sfml-system-3.dll %{cfg.targetdir}",
      }
      links {
         "sfml-graphics",
         "sfml-window",
         "sfml-system",
      }
