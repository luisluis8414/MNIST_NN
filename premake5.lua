workspace "multi_layer_perception"
   architecture "x64"

   configurations {
      "Debug", 
      "Release" 
   }


project "multi_layer_perception"
   kind "ConsoleApp"
   language "C++"
   cppdialect "C++17"
   staticruntime "on"

   targetdir "bin/%{cfg.platform}/%{cfg.buildcfg}"
   objdir "obj/%{cfg.platform}/%{cfg.buildcfg}"

   files { "src/**.h", "src/**.cpp", "src/**.hpp" }

   includedirs { "include" }

   libdirs { "lib" }



   filter "system:windows"
      systemversion "latest"

      defines {
         "PLATFORM_WINDOWS"
      }

   filter "configurations:Debug"
      defines "DEBUG"
      runtime "Debug"
      symbols "on"
      postbuildcommands {
         "{COPY} vendor/opencv/build/x64/vc16/bin/opencv_world4110d.dll %{cfg.targetdir}",
         "{COPY} resources %{cfg.targetdir}/resources",
     }
     links {
      "opencv_world4110d"
   }


   filter "configurations:Release"
      defines "NDEBUG"
      runtime "Release"
      optimize "on"
      postbuildcommands {
         "{COPY} vendor/opencv/build/x64/vc16/bin/opencv_world4110.dll %{cfg.targetdir}",
         "{COPY} resources %{cfg.targetdir}/resources",
     }
     links {
      "opencv_world4110"
   }
