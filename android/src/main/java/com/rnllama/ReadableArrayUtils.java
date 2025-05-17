package com.rnllama;

import com.facebook.react.bridge.ReadableArray;
import com.facebook.react.bridge.ReadableType;

public class ReadableArrayUtils {
    /**
     * Converts a ReadableArray of strings to a semicolon-separated string
     * Used for sending multiple image paths to native code
     */
    public static String stringArrayToSemicolonString(ReadableArray array) {
        if (array == null) {
            return null;
        }

        StringBuilder sb = new StringBuilder();

        for (int i = 0; i < array.size(); i++) {
            if (array.getType(i) == ReadableType.String) {
                if (sb.length() > 0) {
                    sb.append(";");
                }
                sb.append(array.getString(i));
            }
        }

        return sb.length() > 0 ? sb.toString() : null;
    }
}
