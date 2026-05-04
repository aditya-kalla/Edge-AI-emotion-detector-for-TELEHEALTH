from collections import deque, Counter

class SentryEmotionFilter:
    def __init__(self, window_size=6):
        # Rolling memory of last 6 frames (~0.2s)
        self.history = deque(maxlen=window_size)
        self.current_state = "Neutral"

    def update(self, new_prediction, confidence):
        # Accept lower confidence to catch subtle micro-expressions
        if confidence > 0.25:
            self.history.append(new_prediction)
        else:
            self.history.append("Neutral")

        if len(self.history) < self.history.maxlen:
            return new_prediction

        counts = Counter(self.history)

        # TRIGGER 1: Contempt Micro-expression Protection[cite: 4]
        if counts["Contempt"] >= 1:
            self.current_state = "Contempt"
            return self.current_state

        # TRIGGER 2: Anti-Neutral Bias for Sadness
        if counts["Sad"] >= 2:
            self.current_state = "Sad"
            return self.current_state

        # TRIGGER 3: Anger Semantic Priority (Corrects Happy/Anger conflict)[cite: 1]
        if counts["Anger"] >= 2 and counts["Happy"] > 0:
            self.current_state = "Anger"
            return self.current_state

        # Fallback to Mode (most frequent)
        self.current_state = counts.most_common(1)[0][0]
        return self.current_state