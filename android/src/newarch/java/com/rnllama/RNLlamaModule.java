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
  public void toggleNativeLog(boolean enabled, Promise promise) {
    rnllama.toggleNativeLog(enabled, promise);
  }

  @ReactMethod
  public void setContextLimit(double limit, Promise promise) {
    rnllama.setContextLimit(limit, promise);
  }

  @ReactMethod
  public void modelInfo(final String model, final ReadableArray skip, final Promise promise) {
    rnllama.modelInfo(model, skip, promise);
  }

  @ReactMethod
  public void initContext(double id, final ReadableMap params, final Promise promise) {
    rnllama.initContext(id, params, promise);
  }

  @ReactMethod
  public void initMultimodal(double id, final ReadableMap params, final Promise promise) {
    rnllama.initMultimodal(id, params, promise);
  }

  @ReactMethod
  public void isMultimodalEnabled(double id, final Promise promise) {
    rnllama.isMultimodalEnabled(id, promise);
  }

  @ReactMethod
  public void getMultimodalSupport(double id, final Promise promise) {
    rnllama.getMultimodalSupport(id, promise);
  }

  @ReactMethod
  public void releaseMultimodal(double id, final Promise promise) {
    rnllama.releaseMultimodal(id, promise);
  }

  @ReactMethod
  public void getFormattedChat(double id, String messages, String chatTemplate, ReadableMap params, Promise promise) {
    rnllama.getFormattedChat(id, messages, chatTemplate, params, promise);
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
  public void tokenize(double id, final String text, final ReadableArray media_paths, final Promise promise) {
    rnllama.tokenize(id, text, media_paths, promise);
  }

  @ReactMethod
  public void detokenize(double id, final ReadableArray tokens, final Promise promise) {
    rnllama.detokenize(id, tokens, promise);
  }

  @ReactMethod
  public void embedding(double id, final String text, final ReadableMap params, final Promise promise) {
    rnllama.embedding(id, text, params, promise);
  }

  @ReactMethod
  public void rerank(double id, final String query, final ReadableArray documents, final ReadableMap params, final Promise promise) {
    rnllama.rerank(id, query, documents, params, promise);
  }

  @ReactMethod
  public void bench(double id, final double pp, final double tg, final double pl, final double nr, final Promise promise) {
    rnllama.bench(id, pp, tg, pl, nr, promise);
  }

  @ReactMethod
  public void applyLoraAdapters(double id, final ReadableArray loraAdapters, final Promise promise) {
    rnllama.applyLoraAdapters(id, loraAdapters, promise);
  }

  @ReactMethod
  public void removeLoraAdapters(double id, final Promise promise) {
    rnllama.removeLoraAdapters(id, promise);
  }

  @ReactMethod
  public void getLoadedLoraAdapters(double id, final Promise promise) {
    rnllama.getLoadedLoraAdapters(id, promise);
  }

  @ReactMethod
  public void initVocoder(double id, final String vocoderModelPath, final Promise promise) {
    rnllama.initVocoder(id, vocoderModelPath, promise);
  }

  @ReactMethod
  public void isVocoderEnabled(double id, final Promise promise) {
    rnllama.isVocoderEnabled(id, promise);
  }

  @ReactMethod
  public void getFormattedAudioCompletion(double id, final String speakerJsonStr, final String textToSpeak, final Promise promise) {
    rnllama.getFormattedAudioCompletion(id, speakerJsonStr, textToSpeak, promise);
  }

  @ReactMethod
  public void getAudioCompletionGuideTokens(double id, final String textToSpeak, final Promise promise) {
    rnllama.getAudioCompletionGuideTokens(id, textToSpeak, promise);
  }

  @ReactMethod
  public void decodeAudioTokens(double id, final ReadableArray tokens, final Promise promise) {
    rnllama.decodeAudioTokens(id, tokens, promise);
  }

  @ReactMethod
  public void releaseVocoder(double id, final Promise promise) {
    rnllama.releaseVocoder(id, promise);
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
