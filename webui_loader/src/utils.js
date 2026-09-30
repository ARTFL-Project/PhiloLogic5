export function formatSize(bytes) {
    if (bytes === null || bytes === undefined) {
        return "";
    }
    const units = ["B", "KB", "MB", "GB", "TB"];
    let size = bytes;
    let unit = 0;
    while (size >= 1024 && unit < units.length - 1) {
        size /= 1024;
        unit += 1;
    }
    return unit === 0 ? `${size} ${units[unit]}` : `${size.toFixed(1)} ${units[unit]}`;
}

export function formatDate(seconds, locale) {
    if (!seconds) {
        return "";
    }
    return new Date(seconds * 1000).toLocaleString(locale);
}

export function formatDuration(seconds) {
    if (seconds === null || seconds === undefined || seconds < 0) {
        return "";
    }
    const total = Math.round(seconds);
    const hours = Math.floor(total / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    const rest = total % 60;
    if (hours) {
        return `${hours} h ${String(minutes).padStart(2, "0")} min`;
    }
    if (minutes) {
        return `${minutes} min ${String(rest).padStart(2, "0")} s`;
    }
    return `${rest} s`;
}

export function clone(value) {
    return value === undefined ? undefined : JSON.parse(JSON.stringify(value));
}

export function deepEqual(a, b) {
    return JSON.stringify(a) === JSON.stringify(b);
}
