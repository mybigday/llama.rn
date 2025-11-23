package com.rnllama;

import androidx.annotation.NonNull;

import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.module.annotations.ReactModule;
import com.facebook.react.turbomodule.core.CallInvokerHolderImpl;

@ReactModule(name = RNLlama.NAME)
public class RNLlamaModule extends NativeRNLlamaSpec {
  public static final String NAME = RNLlama.NAME;

  public RNLlamaModule(ReactApplicationContext reactContext) {
    super(reactContext);
  }

  @Override
  @NonNull
  public String getName() {
    return NAME;
  }

  @Override
  public void install(Promise promise) {
    boolean result = RNLlamaModuleShared.installJSI(
      getReactApplicationContext(),
      this::installJSIBindings
    );
    promise.resolve(result);
  }

  private native void installJSIBindings(long jsContextPointer, CallInvokerHolderImpl callInvokerHolder);
  private native void cleanupJSIBindings();

  @Override
  public void invalidate() {
    try {
      cleanupJSIBindings();
    } catch (UnsatisfiedLinkError ignored) {
      // Native library may not be loaded if install was never called.
    }
    super.invalidate();
  }
}
