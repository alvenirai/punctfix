from typing import List, Optional

from punctfix.inference import PunctFixer, WordPrediction


class PunctFixStreamer:
    """
    A stateful streamer that receives text in segments, on-line performing punct-fixing and
    returning partial results during streaming. These partial results are guaranteed to be
    final.
    """

    chunked_words: List[WordPrediction]
    buffer: List[WordPrediction]

    def __init__(self, punct_fixer: PunctFixer):
        """
        Takes in an instantiated punct fixer.
        """
        self.punct_fixer = punct_fixer
        self.clear()

    def __call__(self, new_text_segment: str) -> Optional[str]:
        """
        Stream in new text, returning None if this new text did not change anything
        and the partial, finalized text if there has been updates to it.
        """
        self.buffer.extend(
            self.punct_fixer.init_word_prediction_list(
                self.punct_fixer.split_input_text(new_text_segment)
            )
        )
        if self.process_buffer():
            return self.get_result()
        return None

    def finalize(self):
        """
        Mark end of stream and return final puncatuated string.
        """
        self.process_buffer(is_finalized=True)
        punctuated = self.get_result(is_finalized=True)
        self.clear()
        return punctuated

    def get_result(self, is_finalized=False) -> str:
        """
        Returns punctuated string in of all inputs streamed in so far.
        If called when not finalized, will only return text that is certain/no longer subject to change
        """
        if is_finalized:
            finalized_words = self.chunked_words
        # These lines perform a tricky calculation in a dumb way:
        # When is each word finalized? When it has gotten all the labels that it will get.
        # This number of labels is not constant across the sequence and depends on overlap
        # size and on chunk size. To avoid trying to be clever, I just calculate the chunks
        # and overlaps and sum up how many times each index will be in a chunk.
        else:
            # The + chunk size makes calculation takes into account that there will be more
            # chunks in future and that we should not finalize prematurely
            final_num_preds = [0] * (
                len(self.chunked_words) + self.punct_fixer.word_chunk_size
            )
            for chunk in self.punct_fixer.split_words_into_chunks(
                range(len(self.chunked_words))
            ):
                for idx in chunk:
                    final_num_preds[idx] += 1
            finalized_words = [
                word
                for i, word in enumerate(self.chunked_words)
                if len(word.labels) == final_num_preds[i]
            ]
        return self.punct_fixer.combine_word_predictions_into_final_text(
            finalized_words
        )

    def process_buffer(self, is_finalized=False) -> bool:
        """
        Performs actual punctfixing of content in buffer, updating internal state such that a maximal number
        of words get predicted labels. Returns true if new chunks were created and processed and false if not.
        """
        new_chunks = []
        # Save how many words were chunked before this call
        this_processing_started_at = (
            len(self.chunked_words) - self.punct_fixer.word_overlap
            if self.chunked_words
            else 0
        )
        # Whole chunks are appended unless the stream is finalized in which case, the buffer
        # is completely emptied
        while len(self.buffer) >= self.punct_fixer.word_chunk_size or (
            is_finalized and self.buffer
        ):
            new_chunks.append(
                [word.word for word in self.buffer[: self.punct_fixer.word_chunk_size]]
            )
            # Not all words are chunked for the first time, we must (except for first time)
            # skip the first `word_overlap` words to avoid duplicates.
            already_chunked_idx = (
                self.punct_fixer.word_overlap if self.chunked_words else 0
            )
            self.chunked_words.extend(
                self.buffer[already_chunked_idx : self.punct_fixer.word_chunk_size]
            )
            # We don't remove the entire buffer length from the buffer as we want
            # to emulate the overlap feature of the punctfixer; we leave some in there for next chunk.
            self.buffer = self.buffer[
                self.punct_fixer.word_chunk_size - self.punct_fixer.word_overlap :
            ]
        if new_chunks:
            # Run the forward pass on all new chunks, matching with the words that are included in them
            self.punct_fixer.populate_word_prediction_with_labels(
                new_chunks, self.chunked_words[this_processing_started_at:]
            )
            return True
        return False

    def clear(self):
        """
        Reset internal state.
        """
        self.buffer = []
        self.chunked_words = []
