package com.rnllama;

import androidx.annotation.NonNull;

import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.module.annotations.ReactModule;
import com.facebook.react.turbomodule.core.CallInvokerHolderImpl;

@ReactModule(name = RNLlama.NAME)
public class RNLlamaModule extends NativeRNLlamaSpec {
  public static final String NAME = RNLlama.NAME;

  private ReactApplicationContext context;

  public RNLlamaModule(ReactApplicationContext reactContext) {
    super(reactContext);
    this.context = reactContext;
  }

  @Override
  @NonNull
  public String getName() {
    return NAME;
  }

  @Override
  public void install(Promise promise) {
    try {
      boolean loaded = RNLlama.loadNative(context);
      if (!loaded) {
        promise.resolve(false);
        return;
      }

      long jsContextPointer = context.getJavaScriptContextHolder().get();
      CallInvokerHolderImpl holder =
        (CallInvokerHolderImpl) context.getCatalystInstance().getJSCallInvokerHolder();

      if (jsContextPointer == 0 || holder == null) {
        promise.resolve(false);
        return;
      }

      installJSIBindings(jsContextPointer, holder);
      promise.resolve(true);
      return;
    } catch (UnsatisfiedLinkError e) {
    } catch (Exception e) {
    }
    promise.resolve(false);
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
