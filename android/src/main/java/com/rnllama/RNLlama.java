package com.rnllama;

import android.os.Build;
import android.util.Log;

import com.facebook.react.bridge.ReactApplicationContext;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.InputStream;
import java.io.IOException;
import java.util.Locale;
import java.util.regex.Pattern;

public class RNLlama {
  public static final String NAME = "RNLlama";
  private static final String TAG = "RNLlama";
  private static boolean libsLoaded = false;

  // HTP (Hexagon Tensor Processor) constants
  private static final int HTP_DIR_MODE = 0755;  // rwx for owner, rx for group/others
  private static final int HTP_FILE_MODE = 0755;
  private static final String HTP_DIR_NAME = "rnllama-htp";
  private static final String[] HTP_LIBS = {
    "libggml-htp-v69.so",
    "libggml-htp-v73.so",
    "libggml-htp-v75.so",
    "libggml-htp-v79.so",
    "libggml-htp-v81.so"
  };

  private final ReactApplicationContext reactContext;

  public RNLlama(ReactApplicationContext reactContext) {
    this.reactContext = reactContext;
  }

  // HTP library extraction and setup
  private static boolean prepareHtpDirectory(File dir, String label) {
    if (dir == null) {
      return false;
    }
    try {
      if (dir.exists()) {
        if (!dir.isDirectory()) {
          Log.w(NAME, label + " exists but is not a directory: " + dir.getAbsolutePath());
          return false;
        }
      } else {
        if (!dir.mkdirs()) {
          Log.w(NAME, "Unable to create " + label + " at " + dir.getAbsolutePath());
          return false;
        }
        File sanity = File.createTempFile("htp", ".tmp", dir);
        sanity.delete();
      }
    } catch (Exception e) {
      Log.w(NAME, "Unable to prepare " + label + " at " + dir.getAbsolutePath(), e);
      return false;
    }

    dir.setReadable(true, false);
    dir.setExecutable(true, false);
    dir.setWritable(true, true);

    try {
      android.system.Os.chmod(dir.getAbsolutePath(), HTP_DIR_MODE);
    } catch (Exception e) {
      Log.w(NAME, "Failed to chmod HTP directory " + dir.getAbsolutePath(), e);
    }

    return true;
  }

  private static File getPrivateHtpDir(android.content.Context context) {
    try {
      return context.getDir(HTP_DIR_NAME, android.content.Context.MODE_PRIVATE);
    } catch (Exception e) {
      Log.w(NAME, "Unable to access private HTP directory", e);
      return null;
    }
  }

  private static File resolveHtpDirectory(android.content.Context context) {
    File[] candidates = new File[] {
      getPrivateHtpDir(context),
      new File(context.getFilesDir(), HTP_DIR_NAME),
      context.getCodeCacheDir() != null ? new File(context.getCodeCacheDir(), HTP_DIR_NAME) : null,
      context.getCacheDir() != null ? new File(context.getCacheDir(), HTP_DIR_NAME) : null,
      context.getExternalFilesDir(null) != null ? new File(context.getExternalFilesDir(null), HTP_DIR_NAME) : null
    };

    for (File candidate : candidates) {
      if (candidate == null) continue;
      if (prepareHtpDirectory(candidate, "HTP directory candidate")) {
        return candidate;
      }
    }

    Log.w(NAME, "Unable to provision directory for Hexagon libraries; Hexagon backend will be disabled");
    return null;
  }

  private static void setHtpFilePermissions(File file) {
    file.setReadable(true, false);
    file.setExecutable(true, false);
    try {
      android.system.Os.chmod(file.getAbsolutePath(), HTP_FILE_MODE);
    } catch (Exception e) {
      Log.w(NAME, "Failed to chmod HTP library " + file.getAbsolutePath(), e);
    }
  }

  private static boolean ensureHtpLibraries(android.content.Context context, File htpDir) {
    for (String libName : HTP_LIBS) {
      File outFile = new File(htpDir, libName);

      if (outFile.exists()) {
        continue;
      }

      try {
        try (InputStream in = context.getAssets().open("ggml-hexagon/" + libName);
             FileOutputStream out = new FileOutputStream(outFile)) {
          byte[] buffer = new byte[8192];
          int read;
          while ((read = in.read(buffer)) != -1) {
            out.write(buffer, 0, read);
          }
          out.flush();
        }

        setHtpFilePermissions(outFile);
        Log.d(NAME, "Extracted HTP library: " + libName);
      } catch (Exception e) {
        Log.w(NAME, "Failed to extract HTP library " + libName, e);
        return false;
      }
    }
    return true;
  }

  private static void extractHtpLibrariesFromAssets(android.content.Context context) {
    File htpDir = resolveHtpDirectory(context);
    if (htpDir == null) {
      Log.w(NAME, "Could not resolve HTP directory; Hexagon backend will be disabled");
      return;
    }

    Log.d(NAME, "Using " + htpDir.getAbsolutePath() + " for HTP libraries");

    if (!ensureHtpLibraries(context, htpDir)) {
      Log.w(NAME, "Could not install Hexagon libraries; Hexagon backend will be disabled");
      return;
    }

    try {
      String htpLibPath = htpDir.getAbsolutePath();
      android.system.Os.setenv("ADSP_LIBRARY_PATH", htpLibPath, true);
      android.system.Os.setenv("LM_GGML_HEXAGON_NDEV", "16", true);
      Log.d(NAME, "Set ADSP_LIBRARY_PATH=" + htpLibPath);
    } catch (Exception e) {
      Log.w(NAME, "Failed to set ADSP_LIBRARY_PATH", e);
    }
  }

  public static synchronized boolean loadNative(ReactApplicationContext context) {
    if (libsLoaded) return true;

    if (Build.SUPPORTED_64_BIT_ABIS.length == 0) {
      Log.w(TAG, "Only 64-bit architectures are supported");
      return false;
    }

    // Extract HTP libraries from assets before loading native library
    try {
      extractHtpLibrariesFromAssets(context);
    } catch (Exception e) {
      Log.w(TAG, "Failed to extract HTP libraries", e);
    }

    String cpuFeatures = getCpuFeatures();
    boolean hasFp16 = cpuFeatures.contains("fp16") || cpuFeatures.contains("fphp");
    boolean hasDotProd = cpuFeatures.contains("dotprod") || cpuFeatures.contains("asimddp");
    boolean hasI8mm = cpuFeatures.contains("i8mm");

    String hwInfo = (Build.HARDWARE + " " + Build.MANUFACTURER + " " + Build.MODEL).toLowerCase(Locale.ROOT);
    boolean hasAdreno = Pattern.compile("(adreno|qcom|qualcomm)").matcher(hwInfo).find();
    boolean hasHexagon = isHexagonSupported();

    try {
      boolean jniLoaded = false;
      String loadedLib = "";
      if (isArm64V8a()) {
        if (hasDotProd && hasI8mm && hasHexagon && hasAdreno) {
          try { System.loadLibrary("rnllama_jni_v8_2_dotprod_i8mm_hexagon_opencl"); jniLoaded = true; loadedLib = "rnllama_jni_v8_2_dotprod_i8mm_hexagon_opencl"; } catch (UnsatisfiedLinkError ignored) {}
        } else if (hasDotProd && hasI8mm) {
          try { System.loadLibrary("rnllama_jni_v8_2_dotprod_i8mm"); jniLoaded = true; loadedLib = "rnllama_jni_v8_2_dotprod_i8mm"; } catch (UnsatisfiedLinkError ignored) {}
        } else if (hasDotProd) {
          try { System.loadLibrary("rnllama_jni_v8_2_dotprod"); jniLoaded = true; loadedLib = "rnllama_jni_v8_2_dotprod"; } catch (UnsatisfiedLinkError ignored) {}
        } else if (hasI8mm) {
          try { System.loadLibrary("rnllama_jni_v8_2_i8mm"); jniLoaded = true; loadedLib = "rnllama_jni_v8_2_i8mm"; } catch (UnsatisfiedLinkError ignored) {}
        } else if (hasFp16) {
          try { System.loadLibrary("rnllama_jni_v8_2"); jniLoaded = true; loadedLib = "rnllama_jni_v8_2"; } catch (UnsatisfiedLinkError ignored) {}
        } else {
          try { System.loadLibrary("rnllama_jni_v8"); jniLoaded = true; loadedLib = "rnllama_jni_v8"; } catch (UnsatisfiedLinkError ignored) {}
        }
      } else if (isX86_64()) {
        try { System.loadLibrary("rnllama_jni_x86_64"); jniLoaded = true; loadedLib = "rnllama_jni_x86_64"; } catch (UnsatisfiedLinkError ignored) {}
      } else {
        try { System.loadLibrary("rnllama_jni"); jniLoaded = true; loadedLib = "rnllama_jni"; } catch (UnsatisfiedLinkError ignored) {}
      }

      if (!jniLoaded) {
        // Fallback to generic JNI
        System.loadLibrary("rnllama_jni");
        loadedLib = "rnllama_jni";
      }

      System.loadLibrary("rnllama");
      nativeSetLoadedLibrary(loadedLib);
      libsLoaded = true;
      return true;
    } catch (UnsatisfiedLinkError e) {
      Log.e(TAG, "Failed to load native libraries", e);
      return false;
    }
  }

  private static native void nativeSetLoadedLibrary(String name);

  private static boolean isArm64V8a() {
    for (String abi : Build.SUPPORTED_ABIS) {
      if ("arm64-v8a".equalsIgnoreCase(abi)) return true;
    }
    return false;
  }

  private static boolean isX86_64() {
    for (String abi : Build.SUPPORTED_ABIS) {
      if ("x86_64".equalsIgnoreCase(abi)) return true;
    }
    return false;
  }

  private static String getCpuFeatures() {
    StringBuilder features = new StringBuilder();
    try (BufferedReader br = new BufferedReader(new FileReader("/proc/cpuinfo"))) {
      String line;
      while ((line = br.readLine()) != null) {
        String lower = line.toLowerCase(Locale.ROOT);
        if (lower.startsWith("features") || lower.startsWith("flags")) {
          int idx = lower.indexOf(':');
          if (idx != -1 && idx + 1 < lower.length()) {
            features.append(lower.substring(idx + 1).trim()).append(" ");
          }
        }
      }
    } catch (IOException ignored) {
    }
    return features.toString();
  }

  private static boolean isHexagonSupported() {
    if (Build.VERSION.SDK_INT >= 31) {
      String socModel = Build.SOC_MODEL;
      if (socModel != null) {
        socModel = socModel.toUpperCase(Locale.ROOT);
        if (socModel.matches(".*(SM8450|SM8550|SM8650|SM8635|SM8750).*")) {
          return true;
        }
      }
    }
    String hardware = Build.HARDWARE.toLowerCase(Locale.ROOT);
    String board = Build.BOARD.toLowerCase(Locale.ROOT);
    return hardware.matches(".*(taro|kalama|pineapple|sun|lanai).*") ||
      board.matches(".*(taro|kalama|pineapple|sun|lanai).*");
  }
}
