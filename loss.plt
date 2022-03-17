set terminal wxt size 500,500 enhanced font 'Verdana,10' persist

set xlabel 'Epochs'
set ylabel 'Cross-entropy Loss'
set xrange [1:1 < * < 100]
set yrange [0:1 < * < 1000]

plot 'data/loss_data.dat' title 'Loss' with lines
