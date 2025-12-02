
ROOT="$1"

if [ -z "$ROOT" ]; then
    echo "Errore: specify a path."
    exit 1
fi

find "$ROOT" -type d -print0 | while IFS= read -r -d '' DIR; do
    # Controlla se esiste almeno un file .log nella cartella
    if ! find "$DIR" -maxdepth 1 -type f -name "*.log" | grep -q . ; then
        # Evita di eliminare la root stessa
        if [ "$DIR" != "$ROOT" ]; then
            echo "Removing: $DIR"
            rm -rf "$DIR"
        fi
    fi
done