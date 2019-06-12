t = 0:0.001:1-0.001;
x = cos(2*pi*100*t);
figure
acf(x',50)
