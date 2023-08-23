package com.rnllama;

import androidx.annotation.NonNull;
import android.util.Log;
import android.os.Build;
import android.os.Handler;
import android.os.AsyncTask;

import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.LifecycleEventListener;
import com.facebook.react.bridge.ReadableMap;
import com.facebook.react.bridge.WritableMap;
import com.facebook.react.module.annotations.ReactModule;

import java.util.HashMap;
import java.util.Random;
import java.io.File;
import java.io.FileInputStream;
import java.io.PushbackInputStream;

@ReactModule(name = RNLlamaModule.NAME)
public class RNLlamaModule extends ReactContextBaseJavaModule implements LifecycleEventListener {
  public static final String NAME = "RNLlama";

  private ReactApplicationContext reactContext;

  public RNLlamaModule(ReactApplicationContext reactContext) {
    super(reactContext);
    reactContext.addLifecycleEventListener(this);
    this.reactContext = reactContext;
  }

  @Override
  @NonNull
  public String getName() {
    return NAME;
  }

  private HashMap<Integer, LlamaContext> contexts = new HashMap<>();

  @ReactMethod
  public void initContext(final ReadableMap options, final Promise promise) {
    new AsyncTask<Void, Void, Integer>() {
      private Exception exception;

      @Override
      protected Integer doInBackground(Void... voids) {
        try {
          String modelPath = options.getString("filePath");
          String modelFilePath = modelPath;

          long context = LlamaContext.initContext(modelFilePath);
          if (context == 0) {
            throw new Exception("Failed to initialize context");
          }
          int id = Math.abs(new Random().nextInt());
          LlamaContext llamaContext = new LlamaContext(id, reactContext, context);
          contexts.put(id, llamaContext);
          return id;
        } catch (Exception e) {
          exception = e;
          return null;
        }
      }

      @Override
      protected void onPostExecute(Integer id) {
        if (exception != null) {
          promise.reject(exception);
          return;
        }
        promise.resolve(id);
      }
    }.execute();
  }

  @ReactMethod
  public void setContextLimit(double limit, Promise promise) {
    // TODO: implement
  }

  @ReactMethod
  public void completion(double contextId, final ReadableMap options, final Promise promise) {
    // TODO: implement
  }

  @ReactMethod
  public void stopCompletion(double contextId, final Promise promise) {
    // TODO: implement
  }

  @ReactMethod
  public void tokenize(double contextId, final String text, final Promise promise) {
    // TODO: implement
  }

  @ReactMethod
  public void embedding(double contextId, final String text, final Promise promise) {
    // TODO: implement
  }

  @ReactMethod
  public void releaseContext(double id, Promise promise) {
    final int contextId = (int) id;
    new AsyncTask<Void, Void, Void>() {
      private Exception exception;

      @Override
      protected Void doInBackground(Void... voids) {
        try {
          LlamaContext context = contexts.get(contextId);
          if (context == null) {
            throw new Exception("Context " + id + " not found");
          }
          context.release();
          contexts.remove(contextId);
        } catch (Exception e) {
          exception = e;
        }
        return null;
      }

      @Override
      protected void onPostExecute(Void result) {
        if (exception != null) {
          promise.reject(exception);
          return;
        }
        promise.resolve(null);
      }
    }.execute();
  }

  @ReactMethod
  public void releaseAllContexts(Promise promise) {
    new AsyncTask<Void, Void, Void>() {
      private Exception exception;

      @Override
      protected Void doInBackground(Void... voids) {
        try {
          onHostDestroy();
        } catch (Exception e) {
          exception = e;
        }
        return null;
      }

      @Override
      protected void onPostExecute(Void result) {
        if (exception != null) {
          promise.reject(exception);
          return;
        }
        promise.resolve(null);
      }
    }.execute();
  }

  @Override
  public void onHostResume() {
  }

  @Override
  public void onHostPause() {
  }

  @Override
  public void onHostDestroy() {
    for (LlamaContext context : contexts.values()) {
      context.release();
    }
    contexts.clear();
  }
}
