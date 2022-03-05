set terminal wxt size 500,500 enhanced font 'Verdana,10' persist

set xlabel 'False Positive Rate (FPR)'
set ylabel 'True Positive Rate (TPR)'
set xrange [0:1]
set yrange [0:1]

f(x) = x
plot 'Debug/roc_data.txt' title 'ROC' with lines, \
    f(x) title 'TPR = FPR' with lines
