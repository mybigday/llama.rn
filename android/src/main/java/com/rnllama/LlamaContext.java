package com.rnllama;

import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.WritableArray;
import com.facebook.react.bridge.WritableMap;
import com.facebook.react.bridge.ReadableMap;
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
  private int jobId = -1;
  private DeviceEventManagerModule.RCTDeviceEventEmitter eventEmitter;

  private boolean isPredicting = false;
  private boolean isInterrupted = false;

  public LlamaContext(int id, ReactApplicationContext reactContext, ReadableMap params) {
    if (!params.hasKey("model")) {
      throw new IllegalArgumentException("Missing required parameter: model");
    }
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
      // boolean memory_f16,
      params.hasKey("memory_f16") ? params.getBoolean("memory_f16") : true,
      // String lora,
      params.hasKey("lora") ? params.getString("lora") : null,
      // String lora_base,
      params.hasKey("lora_base") ? params.getString("lora_base") : null,
      // float rope_freq_base,
      params.hasKey("rope_freq_base") ? (float) params.getDouble("rope_freq_base") : 10000.0f,
      // float rope_freq_scale
      params.hasKey("rope_freq_scale") ? (float) params.getDouble("rope_freq_scale") : 1.0f
    );
    this.reactContext = reactContext;
    eventEmitter = reactContext.getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter.class);
  }

  public long getContext() {
    return context;
  }

  private void emitPartialCompletion(WritableMap tokenResult) {
    WritableMap event = Arguments.createMap();
    event.putInt("contextId", LlamaContext.this.id);
    event.putMap("tokenResult", tokenResult);
    eventEmitter.emit("@RNLlama_onToken", event);
  }

  private static class PartialCompletionCallback {
    LlamaContext context;

    public PartialCompletionCallback(LlamaContext context) {
      this.context = context;
    }

    void onPartialCompletion(WritableMap tokenResult) {
      context.emitPartialCompletion(tokenResult);
    }
  }

  public WritableMap completion(ReadableMap params) {

    Log.i(NAME, "completion: " + this.context);

    isPredicting = true;
    isInterrupted = false;
    if (!params.hasKey("prompt")) {
      throw new IllegalArgumentException("Missing required parameter: prompt");
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
      // int repeat_last_n,
      params.hasKey("repeat_last_n") ? params.getInt("repeat_last_n") : 64,
      // float repeat_penalty,
      params.hasKey("repeat_penalty") ? (float) params.getDouble("repeat_penalty") : 1.10f,
      // float presence_penalty,
      params.hasKey("presence_penalty") ? (float) params.getDouble("presence_penalty") : 0.00f,
      // float frequency_penalty,
      params.hasKey("frequency_penalty") ? (float) params.getDouble("frequency_penalty") : 0.00f,
      // float mirostat,
      params.hasKey("mirostat") ? (float) params.getDouble("mirostat") : 0.00f,
      // float mirostat_tau,
      params.hasKey("mirostat_tau") ? (float) params.getDouble("mirostat_tau") : 5.00f,
      // float mirostat_eta,
      params.hasKey("mirostat_eta") ? (float) params.getDouble("mirostat_eta") : 0.10f,
      // int top_k,
      params.hasKey("top_k") ? params.getInt("top_k") : 40,
      // float top_p,
      params.hasKey("top_p") ? (float) params.getDouble("top_p") : 0.95f,
      // float tfs_z,
      params.hasKey("tfs_z") ? (float) params.getDouble("tfs_z") : 1.00f,
      // float typical_p,
      params.hasKey("typical_p") ? (float) params.getDouble("typical_p") : 1.00f,
      // String[] stop,
      params.hasKey("stop") ? params.getArray("stop").toArrayList().toArray(new String[0]) : new String[0],
      // boolean ignore_eos,
      params.hasKey("ignore_eos") ? params.getBoolean("ignore_eos") : false,
      // int[][] logit_bias,
      params.hasKey("logit_bias") ? params.getArray("logit_bias").toArrayList().toArray(new int[0][0]) : new int[0][0],
      // PartialCompletionCallback partial_completion_callback
      new PartialCompletionCallback(this)
    );

    isPredicting = false;

    return result;
  }


  public void release() {
    freeContext(context);
  }

  static {
    Log.d(NAME, "Primary ABI: " + Build.SUPPORTED_ABIS[0]);
    boolean loadVfpv4 = false;
    boolean loadV8fp16 = false;
    if (isArmeabiV7a()) {
      // armeabi-v7a needs runtime detection support
      String cpuInfo = cpuInfo();
      if (cpuInfo != null) {
        Log.d(NAME, "CPU info: " + cpuInfo);
        if (cpuInfo.contains("vfpv4")) {
          Log.d(NAME, "CPU supports vfpv4");
          loadVfpv4 = true;
        }
      }
    } else if (isArmeabiV8a()) {
      // ARMv8.2a needs runtime detection support
      String cpuInfo = cpuInfo();
      if (cpuInfo != null) {
        Log.d(NAME, "CPU info: " + cpuInfo);
        if (cpuInfo.contains("fphp")) {
          Log.d(NAME, "CPU supports fp16 arithmetic");
          loadV8fp16 = true;
        }
      }
    }

    if (loadVfpv4) {
      Log.d(NAME, "Loading librnllama_vfpv4.so");
      System.loadLibrary("rnllama_vfpv4");
    } else if (loadV8fp16) {
      Log.d(NAME, "Loading librnllama_v8fp16_va.so");
      System.loadLibrary("rnllama_arm64");
    } else {
      Log.d(NAME, "Loading librnllama.so");
      System.loadLibrary("rnllama");
    }
  }

  private static boolean isArmeabiV7a() {
    return Build.SUPPORTED_ABIS[0].equals("armeabi-v7a");
  }

  private static boolean isArmeabiV8a() {
    return Build.SUPPORTED_ABIS[0].equals("arm64-v8a");
  }

  private static String cpuInfo() {
    File file = new File("/proc/cpuinfo");
    StringBuilder stringBuilder = new StringBuilder();
    try {
      BufferedReader bufferedReader = new BufferedReader(new FileReader(file));
      String line;
      while ((line = bufferedReader.readLine()) != null) {
          stringBuilder.append(line);
      }
      bufferedReader.close();
      return stringBuilder.toString();
    } catch (IOException e) {
      Log.w(NAME, "Couldn't read /proc/cpuinfo", e);
      return null;
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
    boolean memory_f16,
    String lora,
    String lora_base,
    float rope_freq_base,
    float rope_freq_scale
  );
  protected static native WritableMap doCompletion(
    long context_ptr,
    String prompt,
    String grammar,
    float temperature,
    int n_threads,
    int n_predict,
    int n_probs,
    int repeat_last_n,
    float repeat_penalty,
    float presence_penalty,
    float frequency_penalty,
    float mirostat,
    float mirostat_tau,
    float mirostat_eta,
    int top_k,
    float top_p,
    float tfs_z,
    float typical_p,
    String[] stop,
    boolean ignore_eos,
    int[][] logit_bias,
    PartialCompletionCallback partial_completion_callback
  );
  protected static native void freeContext(long contextPtr);
}
