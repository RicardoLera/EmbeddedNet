a=0
for i in $2/*.jpg; do
  new=$(printf "%s/%s%d.jpg" "$2" "$1" "$a") #04 pad to length of 4
  mv -- "$i" "$new"
  ((a++))
done
