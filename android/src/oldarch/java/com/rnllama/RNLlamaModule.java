package com.rnllama;

import androidx.annotation.NonNull;

import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.module.annotations.ReactModule;
import com.facebook.react.turbomodule.core.CallInvokerHolderImpl;

@ReactModule(name = RNLlama.NAME)
public class RNLlamaModule extends ReactContextBaseJavaModule {
  public static final String NAME = RNLlama.NAME;

  public RNLlamaModule(ReactApplicationContext reactContext) {
    super(reactContext);
  }

  @Override
  @NonNull
  public String getName() {
    return NAME;
  }

  @ReactMethod(isBlockingSynchronousMethod = true)
  public boolean install() {
    return RNLlamaModuleShared.installJSI(
      getReactApplicationContext(),
      this::installJSIBindings
    );
  }

  private native void installJSIBindings(long jsContextPointer, CallInvokerHolderImpl callInvokerHolder);
  private native void cleanupJSIBindings();

  @Override
  public void onCatalystInstanceDestroy() {
    try {
      cleanupJSIBindings();
    } catch (UnsatisfiedLinkError ignored) {
      // Native library may not be loaded if install was never called.
    }
    super.onCatalystInstanceDestroy();
  }
}
