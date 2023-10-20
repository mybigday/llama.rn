package com.rnllama;

import androidx.annotation.NonNull;

import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.ReadableMap;
import com.facebook.react.bridge.ReadableArray;
import com.facebook.react.module.annotations.ReactModule;

import java.util.HashMap;
import java.util.Random;
import java.io.File;
import java.io.FileInputStream;
import java.io.PushbackInputStream;

@ReactModule(name = RNLlama.NAME)
public class RNLlamaModule extends NativeRNLlamaSpec {
  public static final String NAME = RNLlama.NAME;

  private RNLlama rnllama = null;

  public RNLlamaModule(ReactApplicationContext reactContext) {
    super(reactContext);
    rnllama = new RNLlama(reactContext);
  }

  @Override
  @NonNull
  public String getName() {
    return NAME;
  }

  @ReactMethod
  public void setContextLimit(double limit, Promise promise) {
    rnllama.setContextLimit(limit, promise);
  }

  @ReactMethod
  public void initContext(final ReadableMap params, final Promise promise) {
    rnllama.initContext(params, promise);
  }

  @ReactMethod
  public void loadSession(double id, String path, Promise promise) {
    rnllama.loadSession(id, path, promise);
  }

  @ReactMethod
  public void saveSession(double id, String path, double size, Promise promise) {
    rnllama.saveSession(id, path, size, promise);
  }

  @ReactMethod
  public void completion(double id, final ReadableMap params, final Promise promise) {
    rnllama.completion(id, params, promise);
  }

  @ReactMethod
  public void stopCompletion(double id, final Promise promise) {
    rnllama.stopCompletion(id, promise);
  }

  @ReactMethod
  public void tokenize(double id, final String text, final Promise promise) {
    rnllama.tokenize(id, text, promise);
  }

  @ReactMethod
  public void detokenize(double id, final ReadableArray tokens, final Promise promise) {
    rnllama.detokenize(id, tokens, promise);
  }

  @ReactMethod
  public void embedding(double id, final String text, final Promise promise) {
    rnllama.embedding(id, text, promise);
  }

  @ReactMethod
  public void releaseContext(double id, Promise promise) {
    rnllama.releaseContext(id, promise);
  }

  @ReactMethod
  public void releaseAllContexts(Promise promise) {
    rnllama.releaseAllContexts(promise);
  }
}
