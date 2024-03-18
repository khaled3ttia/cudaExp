default:
	nvcc -c -o main.o main.cu
	nvcc -c -o kernel.o kernel.cu
	nvcc main.o kernel.o -o blur

clean:
	rm *.o
	rm blur
