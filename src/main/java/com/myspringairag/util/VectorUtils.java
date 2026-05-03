package com.myspringairag.util;

import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;

import java.lang.reflect.Constructor;

/**
 * JVector向量工具类
 */
public class VectorUtils {
    
    private static final VectorTypeSupport VTS;
    
    static {
        try {
            // 通过反射创建ArrayVectorProvider实例（构造函数是包私有的）
            Class<?> providerClass = Class.forName("io.github.jbellis.jvector.vector.ArrayVectorProvider");
            Constructor<?> constructor = providerClass.getDeclaredConstructor();
            constructor.setAccessible(true);
            VTS = (VectorTypeSupport) constructor.newInstance();
        } catch (Exception e) {
            throw new RuntimeException("Failed to create VectorTypeSupport instance", e);
        }
    }

    /**
     * 将float[]转换为VectorFloat
     */
    public static VectorFloat<?> toVectorFloat(float[] vector) {
        return VTS.createFloatVector(vector);
    }
}
