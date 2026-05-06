package com.myspringairag;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableAsync;
import org.springframework.scheduling.annotation.EnableScheduling;

@EnableAsync
@EnableScheduling  // 启用定时任务支持（用于内存监控）
@SpringBootApplication
public class MySpringAIRagApplication {
    
    public static void main(String[] args) {
        SpringApplication.run(MySpringAIRagApplication.class, args);
    }
}
