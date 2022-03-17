set terminal qt size 1000,1000 enhanced font 'Verdana,10' persist
set xlabel 'False Positive Rate (FPR)'
set ylabel 'True Positive Rate (TPR)'
set xrange [0:1]
set yrange [0:1]
set key bottom

if (!exists("n")) n='*'

#ls -1 Debug/roc_data*.dat | sed -e 's/Debug//' -e 's/^\///' -e 's/_data//' -e 's/.dat//'

f(x) = x
FILES = system(sprintf("ls -1 data/roc_data%s.dat", n))
LABEL = system(sprintf("ls -1 data/roc_data%s.dat | sed -e 's/data//' -e 's/_data//' -e 's/.dat//'", n))
plot for [i=1:words(FILES)] word(FILES,i) u 1:2 title word(LABEL,i) noenhanced with lines, \
     f(x) title 'TPR = FPR' with lines lt -1 lw 2 dashtype 2


