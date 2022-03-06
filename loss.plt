set terminal wxt size 500,500 enhanced font 'Verdana,10' persist

set xlabel 'Epochs'
set ylabel 'Cross-entropy Loss'
set xrange [1:1 < * < 10]
set yrange [0:1 < * < 1000]

plot 'Debug/loss_data.txt' title 'Loss' with lines
