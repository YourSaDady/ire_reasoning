for pid in /proc/[0-9]*; do
    if grep -q ":12374" "$pid/net/tcp" 2>/dev/null; then
        echo Port held by PID ${pid##*/}
        ps -p ${pid##*/} -o comm=
        break
    fi
done
echo finished!