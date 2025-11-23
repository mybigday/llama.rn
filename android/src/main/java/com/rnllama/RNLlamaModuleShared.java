package com.rnllama;

import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.turbomodule.core.CallInvokerHolderImpl;

final class RNLlamaModuleShared {
  interface JSIInstaller {
    void install(long jsContextPointer, CallInvokerHolderImpl callInvokerHolder);
  }

  private RNLlamaModuleShared() {}

  static boolean installJSI(ReactApplicationContext context, JSIInstaller installer) {
    try {
      boolean loaded = RNLlama.loadNative(context);
      if (!loaded) {
        return false;
      }

      long jsContextPointer = context.getJavaScriptContextHolder().get();
      CallInvokerHolderImpl holder =
        (CallInvokerHolderImpl) context.getCatalystInstance().getJSCallInvokerHolder();

      if (jsContextPointer == 0 || holder == null) {
        return false;
      }

      installer.install(jsContextPointer, holder);
      return true;
    } catch (UnsatisfiedLinkError e) {
      return false;
    } catch (Exception e) {
      return false;
    }
  }
}
