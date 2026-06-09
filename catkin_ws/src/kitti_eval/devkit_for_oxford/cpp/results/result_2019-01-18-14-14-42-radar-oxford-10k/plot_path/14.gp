set term postscript eps enhanced color
set output "14.eps"
set size ratio -1
set xrange [-618:1432]
set yrange [-861:1189]
set xlabel "x [m]"
set ylabel "y [m]"
plot "14.txt" using 1:2 lc rgb "#FF0000" title 'Ground Truth' w lines,"14.txt" using 3:4 lc rgb "#0000FF" title 'Radar Odometry' w lines,"< head -1 14.txt" using 1:2 lc rgb "#000000" pt 4 ps 1 lw 2 title 'Sequence Start' w points
