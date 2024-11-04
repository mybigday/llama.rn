package com.rnllama;

import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.WritableArray;
import com.facebook.react.bridge.WritableMap;
import com.facebook.react.bridge.ReadableMap;
import com.facebook.react.bridge.ReadableArray;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.modules.core.DeviceEventManagerModule;

import android.util.Log;
import android.os.Build;
import android.content.res.AssetManager;

import java.lang.StringBuilder;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.File;
import java.io.IOException;

public class LlamaContext {
  public static final String NAME = "RNLlamaContext";

  private int id;
  private ReactApplicationContext reactContext;
  private long context;
  private WritableMap modelDetails;
  private int jobId = -1;
  private DeviceEventManagerModule.RCTDeviceEventEmitter eventEmitter;

  public LlamaContext(int id, ReactApplicationContext reactContext, ReadableMap params) {
    if (LlamaContext.isArm64V8a() == false && LlamaContext.isX86_64() == false) {
      throw new IllegalStateException("Only 64-bit architectures are supported");
    }
    if (!params.hasKey("model")) {
      throw new IllegalArgumentException("Missing required parameter: model");
    }
    Log.d(NAME, "Setting log callback");
    logToAndroid();
    eventEmitter = reactContext.getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter.class);
    this.id = id;
    this.context = initContext(
      // String model,
      params.getString("model"),
      // boolean embedding,
      params.hasKey("embedding") ? params.getBoolean("embedding") : false,
      // int n_ctx,
      params.hasKey("n_ctx") ? params.getInt("n_ctx") : 512,
      // int n_batch,
      params.hasKey("n_batch") ? params.getInt("n_batch") : 512,
      // int n_threads,
      params.hasKey("n_threads") ? params.getInt("n_threads") : 0,
      // int n_gpu_layers, // TODO: Support this
      params.hasKey("n_gpu_layers") ? params.getInt("n_gpu_layers") : 0,
      // boolean use_mlock,
      params.hasKey("use_mlock") ? params.getBoolean("use_mlock") : true,
      // boolean use_mmap,
      params.hasKey("use_mmap") ? params.getBoolean("use_mmap") : true,
      //boolean vocab_only,
      params.hasKey("vocab_only") ? params.getBoolean("vocab_only") : false,
      // String lora,
      params.hasKey("lora") ? params.getString("lora") : "",
      // float lora_scaled,
      params.hasKey("lora_scaled") ? (float) params.getDouble("lora_scaled") : 1.0f,
      // float rope_freq_base,
      params.hasKey("rope_freq_base") ? (float) params.getDouble("rope_freq_base") : 0.0f,
      // float rope_freq_scale
      params.hasKey("rope_freq_scale") ? (float) params.getDouble("rope_freq_scale") : 0.0f,
      // LoadProgressCallback load_progress_callback
      params.hasKey("use_progress_callback") ? new LoadProgressCallback(this) : null
    );
    this.modelDetails = loadModelDetails(this.context);
    this.reactContext = reactContext;
  }

  public void interruptLoad() {
    interruptLoad(this.context);
  }

  public long getContext() {
    return context;
  }

  public WritableMap getModelDetails() {
    return modelDetails;
  }

  public String getFormattedChat(ReadableArray messages, String chatTemplate) {
    ReadableMap[] msgs = new ReadableMap[messages.size()];
    for (int i = 0; i < messages.size(); i++) {
      msgs[i] = messages.getMap(i);
    }
    return getFormattedChat(this.context, msgs, chatTemplate == null ? "" : chatTemplate);
  }

  private void emitLoadProgress(int progress) {
    WritableMap event = Arguments.createMap();
    event.putInt("contextId", LlamaContext.this.id);
    event.putInt("progress", progress);
    eventEmitter.emit("@RNLlama_onInitContextProgress", event);
  }

  private static class LoadProgressCallback {
    LlamaContext context;

    public LoadProgressCallback(LlamaContext context) {
      this.context = context;
    }

    void onLoadProgress(int progress) {
      context.emitLoadProgress(progress);
    }
  }

  private void emitPartialCompletion(WritableMap tokenResult) {
    WritableMap event = Arguments.createMap();
    event.putInt("contextId", LlamaContext.this.id);
    event.putMap("tokenResult", tokenResult);
    eventEmitter.emit("@RNLlama_onToken", event);
  }

  private static class PartialCompletionCallback {
    LlamaContext context;
    boolean emitNeeded;

    public PartialCompletionCallback(LlamaContext context, boolean emitNeeded) {
      this.context = context;
      this.emitNeeded = emitNeeded;
    }

    void onPartialCompletion(WritableMap tokenResult) {
      if (!emitNeeded) return;
      context.emitPartialCompletion(tokenResult);
    }
  }

  public WritableMap loadSession(String path) {
    if (path == null || path.isEmpty()) {
      throw new IllegalArgumentException("File path is empty");
    }
    File file = new File(path);
    if (!file.exists()) {
      throw new IllegalArgumentException("File does not exist: " + path);
    }
    WritableMap result = loadSession(this.context, path);
    if (result.hasKey("error")) {
      throw new IllegalStateException(result.getString("error"));
    }
    return result;
  }

  public int saveSession(String path, int size) {
    if (path == null || path.isEmpty()) {
      throw new IllegalArgumentException("File path is empty");
    }
    return saveSession(this.context, path, size);
  }

  public WritableMap completion(ReadableMap params) {
    if (!params.hasKey("prompt")) {
      throw new IllegalArgumentException("Missing required parameter: prompt");
    }

    double[][] logit_bias = new double[0][0];
    if (params.hasKey("logit_bias")) {
      ReadableArray logit_bias_array = params.getArray("logit_bias");
      logit_bias = new double[logit_bias_array.size()][];
      for (int i = 0; i < logit_bias_array.size(); i++) {
        ReadableArray logit_bias_row = logit_bias_array.getArray(i);
        logit_bias[i] = new double[logit_bias_row.size()];
        for (int j = 0; j < logit_bias_row.size(); j++) {
          logit_bias[i][j] = logit_bias_row.getDouble(j);
        }
      }
    }

    WritableMap result = doCompletion(
      this.context,
      // String prompt,
      params.getString("prompt"),
      // String grammar,
      params.hasKey("grammar") ? params.getString("grammar") : "",
      // float temperature,
      params.hasKey("temperature") ? (float) params.getDouble("temperature") : 0.7f,
      // int n_threads,
      params.hasKey("n_threads") ? params.getInt("n_threads") : 0,
      // int n_predict,
      params.hasKey("n_predict") ? params.getInt("n_predict") : -1,
      // int n_probs,
      params.hasKey("n_probs") ? params.getInt("n_probs") : 0,
      // int penalty_last_n,
      params.hasKey("penalty_last_n") ? params.getInt("penalty_last_n") : 64,
      // float penalty_repeat,
      params.hasKey("penalty_repeat") ? (float) params.getDouble("penalty_repeat") : 1.00f,
      // float penalty_freq,
      params.hasKey("penalty_freq") ? (float) params.getDouble("penalty_freq") : 0.00f,
      // float penalty_present,
      params.hasKey("penalty_present") ? (float) params.getDouble("penalty_present") : 0.00f,
      // float mirostat,
      params.hasKey("mirostat") ? (float) params.getDouble("mirostat") : 0.00f,
      // float mirostat_tau,
      params.hasKey("mirostat_tau") ? (float) params.getDouble("mirostat_tau") : 5.00f,
      // float mirostat_eta,
      params.hasKey("mirostat_eta") ? (float) params.getDouble("mirostat_eta") : 0.10f,
      // boolean penalize_nl,
      params.hasKey("penalize_nl") ? params.getBoolean("penalize_nl") : false,
      // int top_k,
      params.hasKey("top_k") ? params.getInt("top_k") : 40,
      // float top_p,
      params.hasKey("top_p") ? (float) params.getDouble("top_p") : 0.95f,
      // float min_p,
      params.hasKey("min_p") ? (float) params.getDouble("min_p") : 0.05f,
      // float xtc_threshold,
      params.hasKey("xtc_threshold") ? (float) params.getDouble("xtc_threshold") : 0.00f,
      // float xtc_probability,
      params.hasKey("xtc_probability") ? (float) params.getDouble("xtc_probability") : 0.00f,
      // float typical_p,
      params.hasKey("typical_p") ? (float) params.getDouble("typical_p") : 1.00f,
      // int seed,
      params.hasKey("seed") ? params.getInt("seed") : -1,
      // String[] stop,
      params.hasKey("stop") ? params.getArray("stop").toArrayList().toArray(new String[0]) : new String[0],
      // boolean ignore_eos,
      params.hasKey("ignore_eos") ? params.getBoolean("ignore_eos") : false,
      // double[][] logit_bias,
      logit_bias,
      // PartialCompletionCallback partial_completion_callback
      new PartialCompletionCallback(
        this,
        params.hasKey("emit_partial_completion") ? params.getBoolean("emit_partial_completion") : false
      )
    );
    if (result.hasKey("error")) {
      throw new IllegalStateException(result.getString("error"));
    }
    return result;
  }

  public void stopCompletion() {
    stopCompletion(this.context);
  }

  public boolean isPredicting() {
    return isPredicting(this.context);
  }

  public WritableMap tokenize(String text) {
    WritableMap result = Arguments.createMap();
    result.putArray("tokens", tokenize(this.context, text));
    return result;
  }

  public String detokenize(ReadableArray tokens) {
    int[] toks = new int[tokens.size()];
    for (int i = 0; i < tokens.size(); i++) {
      toks[i] = (int) tokens.getDouble(i);
    }
    return detokenize(this.context, toks);
  }

  public WritableMap getEmbedding(String text) {
    if (isEmbeddingEnabled(this.context) == false) {
      throw new IllegalStateException("Embedding is not enabled");
    }
    WritableMap result = embedding(this.context, text);
    if (result.hasKey("error")) {
      throw new IllegalStateException(result.getString("error"));
    }
    return result;
  }

  public String bench(int pp, int tg, int pl, int nr) {
    return bench(this.context, pp, tg, pl, nr);
  }

  public void release() {
    freeContext(context);
  }

  static {
    Log.d(NAME, "Primary ABI: " + Build.SUPPORTED_ABIS[0]);

    String cpuFeatures = LlamaContext.getCpuFeatures();
    Log.d(NAME, "CPU features: " + cpuFeatures);
    boolean hasFp16 = cpuFeatures.contains("fp16") || cpuFeatures.contains("fphp");
    boolean hasDotProd = cpuFeatures.contains("dotprod") || cpuFeatures.contains("asimddp");
    boolean hasSve = cpuFeatures.contains("sve");
    boolean hasI8mm = cpuFeatures.contains("i8mm");
    boolean isAtLeastArmV82 = cpuFeatures.contains("asimd") && cpuFeatures.contains("crc32") && cpuFeatures.contains("aes");
    boolean isAtLeastArmV84 = cpuFeatures.contains("dcpop") && cpuFeatures.contains("uscat");
    Log.d(NAME, "- hasFp16: " + hasFp16);
    Log.d(NAME, "- hasDotProd: " + hasDotProd);
    Log.d(NAME, "- hasSve: " + hasSve);
    Log.d(NAME, "- hasI8mm: " + hasI8mm);
    Log.d(NAME, "- isAtLeastArmV82: " + isAtLeastArmV82);
    Log.d(NAME, "- isAtLeastArmV84: " + isAtLeastArmV84);

    // TODO: Add runtime check for cpu features
    if (LlamaContext.isArm64V8a()) {
      if (isAtLeastArmV84 && hasSve && hasI8mm && hasFp16 && hasDotProd) {
        Log.d(NAME, "Loading librnllama_v8_4_fp16_dotprod_i8mm_sve.so");
        System.loadLibrary("rnllama_v8_4_fp16_dotprod_i8mm_sve");
      } else if (isAtLeastArmV84 && hasSve && hasFp16 && hasDotProd) {
        Log.d(NAME, "Loading librnllama_v8_4_fp16_dotprod_sve.so");
        System.loadLibrary("rnllama_v8_4_fp16_dotprod_sve");
      } else if (isAtLeastArmV84 && hasI8mm && hasFp16 && hasDotProd) {
        Log.d(NAME, "Loading librnllama_v8_4_fp16_dotprod_i8mm.so");
        System.loadLibrary("rnllama_v8_4_fp16_dotprod_i8mm");
      } else if (isAtLeastArmV84 && hasFp16 && hasDotProd) {
        Log.d(NAME, "Loading librnllama_v8_4_fp16_dotprod.so");
        System.loadLibrary("rnllama_v8_4_fp16_dotprod");
      } else if (isAtLeastArmV82 && hasFp16 && hasDotProd) {
        Log.d(NAME, "Loading librnllama_v8_2_fp16_dotprod.so");
        System.loadLibrary("rnllama_v8_2_fp16_dotprod");
      } else if (isAtLeastArmV82 && hasFp16) {
        Log.d(NAME, "Loading librnllama_v8_2_fp16.so");
        System.loadLibrary("rnllama_v8_2_fp16");
      } else {
        Log.d(NAME, "Loading librnllama_v8.so");
        System.loadLibrary("rnllama_v8");
      }
      //  Log.d(NAME, "Loading librnllama_v8_7.so with runtime feature detection");
      //  System.loadLibrary("rnllama_v8_7");
    } else if (LlamaContext.isX86_64()) {
        Log.d(NAME, "Loading librnllama_x86_64.so");
        System.loadLibrary("rnllama_x86_64");
    } else {
        Log.d(NAME, "Loading default librnllama.so");
        System.loadLibrary("rnllama");
    }
}

  private static boolean isArm64V8a() {
    return Build.SUPPORTED_ABIS[0].equals("arm64-v8a");
  }

  private static boolean isX86_64() {
    return Build.SUPPORTED_ABIS[0].equals("x86_64");
  }

  private static String getCpuFeatures() {
    File file = new File("/proc/cpuinfo");
    StringBuilder stringBuilder = new StringBuilder();
    try {
      BufferedReader bufferedReader = new BufferedReader(new FileReader(file));
      String line;
      while ((line = bufferedReader.readLine()) != null) {
        if (line.startsWith("Features")) {
          stringBuilder.append(line);
          break;
        }
      }
      bufferedReader.close();
      return stringBuilder.toString();
    } catch (IOException e) {
      Log.w(NAME, "Couldn't read /proc/cpuinfo", e);
      return "";
    }
  }

  protected static native long initContext(
    String model,
    boolean embedding,
    int n_ctx,
    int n_batch,
    int n_threads,
    int n_gpu_layers, // TODO: Support this
    boolean use_mlock,
    boolean use_mmap,
    boolean vocab_only,
    String lora,
    float lora_scaled,
    float rope_freq_base,
    float rope_freq_scale,
    LoadProgressCallback load_progress_callback
  );
  protected static native void interruptLoad(long contextPtr);
  protected static native WritableMap loadModelDetails(
    long contextPtr
  );
  protected static native String getFormattedChat(
    long contextPtr,
    ReadableMap[] messages,
    String chatTemplate
  );
  protected static native WritableMap loadSession(
    long contextPtr,
    String path
  );
  protected static native int saveSession(
    long contextPtr,
    String path,
    int size
  );
  protected static native WritableMap doCompletion(
    long context_ptr,
    String prompt,
    String grammar,
    float temperature,
    int n_threads,
    int n_predict,
    int n_probs,
    int penalty_last_n,
    float penalty_repeat,
    float penalty_freq,
    float penalty_present,
    float mirostat,
    float mirostat_tau,
    float mirostat_eta,
    boolean penalize_nl,
    int top_k,
    float top_p,
    float min_p,
    float xtc_threshold,
    float xtc_probability,
    float typical_p,
    int seed,
    String[] stop,
    boolean ignore_eos,
    double[][] logit_bias,
    PartialCompletionCallback partial_completion_callback
  );
  protected static native void stopCompletion(long contextPtr);
  protected static native boolean isPredicting(long contextPtr);
  protected static native WritableArray tokenize(long contextPtr, String text);
  protected static native String detokenize(long contextPtr, int[] tokens);
  protected static native boolean isEmbeddingEnabled(long contextPtr);
  protected static native WritableMap embedding(long contextPtr, String text);
  protected static native String bench(long contextPtr, int pp, int tg, int pl, int nr);
  protected static native void freeContext(long contextPtr);
  protected static native void logToAndroid();
}
