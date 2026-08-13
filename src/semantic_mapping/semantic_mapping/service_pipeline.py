"""Non-blocking SAM2/RAM++ fan-out followed by bounded SigLIP2 batches."""

import threading
from concurrent.futures import Future
from dataclasses import dataclass

from ibrobot_msgs.srv import EncodeEmbeddings, GenerateMasks, RecognizeTags
from perception_service.model_contracts import MAX_MASK_BATCH


def ram_mask_candidates(
    mask_index: int,
    mask_tag_counts,
    mask_tags,
    mask_scores,
    excluded_labels=(),
) -> tuple[tuple[str, float], ...]:
    if mask_index < 0 or mask_index >= len(mask_tag_counts):
        return ()
    start = sum(int(count) for count in mask_tag_counts[:mask_index])
    stop = start + int(mask_tag_counts[mask_index])
    excluded = {str(value).strip().casefold() for value in excluded_labels}
    candidates = tuple(
        (normalized, float(score))
        for label, score in zip(mask_tags[start:stop], mask_scores[start:stop], strict=True)
        if (normalized := str(label).strip()) and normalized.casefold() not in excluded
    )
    return tuple(sorted(candidates, key=lambda item: (-item[1], item[0].casefold())))


def select_ram_label(
    mask_index: int,
    mask_tag_counts,
    mask_tags,
    mask_scores,
    min_confidence: float,
    excluded_labels=(),
) -> tuple[str, float]:
    """Select the highest-confidence local RAM++ candidate for one mask."""
    candidates = ram_mask_candidates(mask_index, mask_tag_counts, mask_tags, mask_scores, excluded_labels)
    if candidates and candidates[0][1] >= min_confidence:
        return candidates[0]
    return "unlabeled", 0.0


@dataclass(frozen=True)
class ServiceFrameResult:
    masks: object
    tags: tuple[str, ...]
    tag_scores: tuple[float, ...]
    mask_tag_counts: tuple[int, ...]
    mask_tags: tuple[str, ...]
    mask_tag_scores: tuple[float, ...]
    embeddings: tuple
    model_diagnostics: dict[str, tuple]


def split_batches(items, batch_size: int = MAX_MASK_BATCH):
    if batch_size <= 0 or batch_size > MAX_MASK_BATCH:
        raise ValueError(f"batch size must be between 1 and {MAX_MASK_BATCH}")
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


class ServiceFramePipeline:
    """Compose ROS futures; executor threads remain free and no callback spins."""

    def __init__(
        self,
        sam_client,
        ram_client,
        siglip_client,
        *,
        max_masks_per_batch: int = MAX_MASK_BATCH,
        excluded_labels=(),
        max_mask_candidates: int = 5,
    ):
        if not 1 <= max_masks_per_batch <= MAX_MASK_BATCH:
            raise ValueError(f"max_masks_per_batch must be between 1 and {MAX_MASK_BATCH}")
        self.sam_client = sam_client
        self.ram_client = ram_client
        self.siglip_client = siglip_client
        self.max_masks_per_batch = max_masks_per_batch
        self.excluded_labels = tuple(excluded_labels)
        self.max_mask_candidates = max_mask_candidates

    def process(
        self,
        image,
        *,
        mask_options: dict | None = None,
        score_threshold: float = 0.0,
        mask_selector=None,
    ) -> Future:
        mask_options = mask_options or {}
        sam_request = GenerateMasks.Request(image=image, **mask_options)
        ram_request = RecognizeTags.Request(
            image=image,
            masks=[],
            include_image=True,
            score_threshold=score_threshold,
            excluded_labels=[],
            max_mask_candidates=0,
        )
        sam_future = self.sam_client.call_async(sam_request)
        ram_future = self.ram_client.call_async(ram_request)
        result_future = Future()
        state = {"sam": None, "ram": None, "ram_local_started": False}
        state_lock = threading.Lock()

        def fail(exc):
            if not result_future.done():
                result_future.set_exception(exc)

        def stage_one_done(name, future):
            if result_future.done():
                return
            try:
                response = future.result()
                if response is None or not response.success:
                    raise RuntimeError(f"{name} service failed: {getattr(response, 'message', 'no response')}")
                if name == "sam" and mask_selector is not None:
                    detections = response.detections.detections
                    selected = list(mask_selector(detections))
                    if len(selected) != len(set(selected)) or any(
                        index < 0 or index >= len(detections) for index in selected
                    ):
                        raise ValueError("mask selector returned invalid detection indices")
                    response.detections.detections = [detections[index] for index in selected]
                with state_lock:
                    state[name] = response
            except Exception as exc:
                fail(exc)
                return
            with state_lock:
                ready = state["sam"] is not None and state["ram"] is not None and not state["ram_local_started"]
                if ready:
                    state["ram_local_started"] = True
                    sam_response, ram_response = state["sam"], state["ram"]
            if ready:
                self._start_local_ram(image, sam_response, ram_response, result_future, score_threshold)

        sam_future.add_done_callback(lambda future: stage_one_done("sam", future))
        ram_future.add_done_callback(lambda future: stage_one_done("ram", future))
        return result_future

    def _start_siglip(self, image, sam_response, ram_response, result_future: Future) -> None:
        masks = [detection.mask for detection in sam_response.detections.detections]
        local_tags = getattr(ram_response, "mask_tags", ())
        local_scores = getattr(ram_response, "mask_scores", ())
        local_counts = getattr(ram_response, "mask_tag_counts", ())
        if not masks:
            identity_future = self.siglip_client.call_async(
                EncodeEmbeddings.Request(image=image, masks=[], candidate_labels=[])
            )

            def identity_done(future):
                if result_future.done():
                    return
                try:
                    response = future.result()
                    if response is None or not response.success:
                        raise RuntimeError(f"siglip2 service failed: {getattr(response, 'message', 'no response')}")
                    result_future.set_result(
                        ServiceFrameResult(
                            masks=sam_response.detections,
                            tags=tuple(ram_response.tags),
                            tag_scores=tuple(ram_response.scores),
                            mask_tag_counts=tuple(local_counts),
                            mask_tags=tuple(local_tags),
                            mask_tag_scores=tuple(local_scores),
                            embeddings=(),
                            model_diagnostics={
                                "sam2": (sam_response.model,),
                                "ram_plus": (ram_response.model,),
                                "siglip2_image": (response.model,),
                            },
                        )
                    )
                except Exception as exc:
                    result_future.set_exception(exc)

            identity_future.add_done_callback(identity_done)
            return

        requests = split_batches(list(enumerate(masks)), self.max_masks_per_batch)

        batch_futures = []
        for batch in requests:
            request = EncodeEmbeddings.Request(
                image=image,
                masks=[mask for _index, mask in batch],
                candidate_labels=[],
            )
            batch_futures.append(self.siglip_client.call_async(request))
        batch_results = [None] * len(batch_futures)

        def batch_done(index, future):
            if result_future.done():
                return
            try:
                response = future.result()
                if response is None or not response.success:
                    raise RuntimeError(f"siglip2 service failed: {getattr(response, 'message', 'no response')}")
                batch_results[index] = response
            except Exception as exc:
                result_future.set_exception(exc)
                return
            if all(response is not None for response in batch_results):
                embeddings = []
                for response, batch in zip(batch_results, requests, strict=True):
                    for item in response.results:
                        item.mask_index = batch[item.mask_index][0]
                        embeddings.append(item)
                embeddings.sort(key=lambda item: item.mask_index)
                result_future.set_result(
                    ServiceFrameResult(
                        masks=sam_response.detections,
                        tags=tuple(ram_response.tags),
                        tag_scores=tuple(ram_response.scores),
                        mask_tag_counts=tuple(local_counts),
                        mask_tags=tuple(local_tags),
                        mask_tag_scores=tuple(local_scores),
                        embeddings=tuple(embeddings),
                        model_diagnostics={
                            "sam2": (sam_response.model,),
                            "ram_plus": (ram_response.model,),
                            "siglip2_image": tuple(response.model for response in batch_results),
                        },
                    )
                )

        for index, future in enumerate(batch_futures):
            future.add_done_callback(lambda future, index=index: batch_done(index, future))

    def _start_local_ram(self, image, sam_response, ram_response, result_future, score_threshold):
        masks = [detection.mask for detection in sam_response.detections.detections]
        if not masks:
            self._start_siglip(image, sam_response, ram_response, result_future)
            return
        batches = split_batches(masks, self.max_masks_per_batch)
        futures = [
            self.ram_client.call_async(
                RecognizeTags.Request(
                    image=image,
                    masks=batch,
                    include_image=False,
                    score_threshold=score_threshold,
                    excluded_labels=list(self.excluded_labels),
                    max_mask_candidates=self.max_mask_candidates,
                )
            )
            for batch in batches
        ]
        responses = [None] * len(futures)

        def done(index, completed):
            if result_future.done():
                return
            try:
                response = completed.result()
                if response is None or not response.success:
                    raise RuntimeError(f"ram++ mask service failed: {getattr(response, 'message', 'no response')}")
                if len(response.mask_tag_counts) != len(batches[index]):
                    raise RuntimeError("RAM++ returned a different mask-candidate count than the requested batch")
                if sum(response.mask_tag_counts) != len(response.mask_tags) or len(response.mask_tags) != len(
                    response.mask_scores
                ):
                    raise RuntimeError("RAM++ returned inconsistent flattened mask candidates")
                responses[index] = response
                if all(value is not None for value in responses):
                    ram_response.mask_tag_counts = [count for value in responses for count in value.mask_tag_counts]
                    ram_response.mask_tags = [tag for value in responses for tag in value.mask_tags]
                    ram_response.mask_scores = [score for value in responses for score in value.mask_scores]
                    self._start_siglip(image, sam_response, ram_response, result_future)
            except Exception as exc:
                result_future.set_exception(exc)

        for index, future in enumerate(futures):
            future.add_done_callback(lambda completed, index=index: done(index, completed))
