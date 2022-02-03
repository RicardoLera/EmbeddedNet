################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../actions.c \
../blocks.c \
../data_manip.c \
../layers.c \
../main.c \
../operations.c 

OBJS += \
./actions.o \
./blocks.o \
./data_manip.o \
./layers.o \
./main.o \
./operations.o 

C_DEPS += \
./actions.d \
./blocks.d \
./data_manip.d \
./layers.d \
./main.d \
./operations.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.c subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	gcc -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


