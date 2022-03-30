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

C_DEPS += \
./actions.d \
./blocks.d \
./data_manip.d \
./layers.d \
./main.d \
./operations.d 

OBJS += \
./actions.o \
./blocks.o \
./data_manip.o \
./layers.o \
./main.o \
./operations.o 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.c subdir.mk
	@echo 'Building file: $<'
	@echo 'Invoking: Cross GCC Compiler'
	gcc -O3 -mavx2 -Wall -c -fmessage-length=0 -fPIC -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


clean: clean--2e-

clean--2e-:
	-$(RM) ./actions.d ./actions.o ./blocks.d ./blocks.o ./data_manip.d ./data_manip.o ./layers.d ./layers.o ./main.d ./main.o ./operations.d ./operations.o

.PHONY: clean--2e-

