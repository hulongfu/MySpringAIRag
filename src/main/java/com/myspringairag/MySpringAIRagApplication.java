package com.myspringairag;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableAsync;

@EnableAsync
@SpringBootApplication
public class MySpringAIRagApplication {
    
    public static void main(String[] args) {
        SpringApplication.run(MySpringAIRagApplication.class, args);
    }
}
