"use strict";

// Quality-first monocular fallback for devices without WebXR/ARCore.  The
// worker reports one viewer-local displacement per camera frame; the page owns
// the virtual WebXR pose and applies every accepted delta exactly once.
let width = 320;
let height = 240;
let canvas = null;
let context = null;
let previousPyramid = null;
let previousFeatures = [];
let previousOrientation = null;
let previousTimestamp = 0;
let focalLengthPx = 0;
let focalConfidence = 0;

const pyramidLevels = 4;
const maxFeatures = 120;
const featureCellSize = 22;
const lkWindowRadius = 3;
const lkIterations = 6;
const minFeatureScore = 55;
const forwardBackwardLimitPx = 1.75;
const ransacIterations = 48;
const ransacThresholdPx = 2.75;
const minReliableFeatures = 16;
const lateralGainMeters = 0.50;
const depthGainMeters = 0.45;
const lateralDeadzoneNormalized = 0.0008;
const depthDeadzone = 0.0015;
const maxFrameTranslationM = 0.025;
const maxFrameDepthM = 0.012;
const rotationFusionMinRadS = 0.15;
const rotationFusionMinRad = 0.004;
const yawRotationTranslationGainFloor = 0.65;
const tiltRotationTranslationGainFloor = 0.10;
const rotationTranslationFullConfidence = 0.50;

function clamp(value, lower, upper) {
    return Math.max(lower, Math.min(upper, value));
}

function rotationTranslationGain(confidence, gainFloor) {
    // Rotation is already removed geometrically before fitting translation.
    // Confidence therefore softens accepted motion instead of applying a
    // second near-binary penalty that makes simultaneous rotation unusable.
    const normalized = clamp(confidence / rotationTranslationFullConfidence, 0, 1);
    const smooth = normalized * normalized * (3 - 2 * normalized);
    return gainFloor + (1 - gainFloor) * smooth;
}

function grayscale(imageData) {
    const rgba = imageData.data;
    const gray = new Float32Array(width * height);
    for (let index = 0; index < gray.length; index++) {
        const offset = index * 4;
        gray[index] = 0.299 * rgba[offset] + 0.587 * rgba[offset + 1] + 0.114 * rgba[offset + 2];
    }
    return gray;
}

function downsample(source, sourceWidth, sourceHeight) {
    const targetWidth = Math.max(1, Math.floor(sourceWidth / 2));
    const targetHeight = Math.max(1, Math.floor(sourceHeight / 2));
    const target = new Float32Array(targetWidth * targetHeight);
    for (let y = 0; y < targetHeight; y++) {
        for (let x = 0; x < targetWidth; x++) {
            const sourceX = x * 2;
            const sourceY = y * 2;
            const row0 = sourceY * sourceWidth;
            const row1 = Math.min(sourceHeight - 1, sourceY + 1) * sourceWidth;
            const x1 = Math.min(sourceWidth - 1, sourceX + 1);
            target[y * targetWidth + x] = 0.25 * (
                source[row0 + sourceX] + source[row0 + x1] +
                source[row1 + sourceX] + source[row1 + x1]
            );
        }
    }
    return { data: target, width: targetWidth, height: targetHeight };
}

function buildPyramid(gray) {
    const levels = [{ data: gray, width: width, height: height }];
    while (levels.length < pyramidLevels) {
        const previous = levels[levels.length - 1];
        levels.push(downsample(previous.data, previous.width, previous.height));
    }
    return levels;
}

function sampleBilinear(image, imageWidth, imageHeight, x, y) {
    const x0 = Math.floor(x);
    const y0 = Math.floor(y);
    if (x0 < 0 || y0 < 0 || x0 + 1 >= imageWidth || y0 + 1 >= imageHeight) return NaN;
    const dx = x - x0;
    const dy = y - y0;
    const row0 = y0 * imageWidth;
    const row1 = (y0 + 1) * imageWidth;
    const top = image[row0 + x0] * (1 - dx) + image[row0 + x0 + 1] * dx;
    const bottom = image[row1 + x0] * (1 - dx) + image[row1 + x0 + 1] * dx;
    return top * (1 - dy) + bottom * dy;
}

function featureScore(gray, x, y) {
    let gxx = 0;
    let gxy = 0;
    let gyy = 0;
    for (let offsetY = -2; offsetY <= 2; offsetY++) {
        for (let offsetX = -2; offsetX <= 2; offsetX++) {
            const index = (y + offsetY) * width + x + offsetX;
            const gx = 0.5 * (gray[index + 1] - gray[index - 1]);
            const gy = 0.5 * (gray[index + width] - gray[index - width]);
            gxx += gx * gx;
            gxy += gx * gy;
            gyy += gy * gy;
        }
    }
    const trace = gxx + gyy;
    const determinant = gxx * gyy - gxy * gxy;
    return 0.5 * (trace - Math.sqrt(Math.max(0, trace * trace - 4 * determinant)));
}

function detectFeatures(gray) {
    const candidates = [];
    const margin = lkWindowRadius + 5;
    for (let cellY = margin; cellY < height - margin; cellY += featureCellSize) {
        for (let cellX = margin; cellX < width - margin; cellX += featureCellSize) {
            let best = null;
            const endY = Math.min(height - margin, cellY + featureCellSize);
            const endX = Math.min(width - margin, cellX + featureCellSize);
            for (let y = cellY; y < endY; y += 2) {
                for (let x = cellX; x < endX; x += 2) {
                    const score = featureScore(gray, x, y);
                    if (score >= minFeatureScore && (!best || score > best.score)) {
                        best = { x: x, y: y, score: score };
                    }
                }
            }
            if (best) candidates.push(best);
        }
    }
    candidates.sort((left, right) => right.score - left.score);
    return candidates.slice(0, maxFeatures).map((point) => ({ x: point.x, y: point.y }));
}

function trackPointAtLevel(source, target, sourcePoint, initialTargetPoint) {
    let targetX = initialTargetPoint.x;
    let targetY = initialTargetPoint.y;
    const margin = lkWindowRadius + 2;
    for (let iteration = 0; iteration < lkIterations; iteration++) {
        if (
            sourcePoint.x < margin || sourcePoint.y < margin ||
            sourcePoint.x >= source.width - margin || sourcePoint.y >= source.height - margin ||
            targetX < margin || targetY < margin ||
            targetX >= target.width - margin || targetY >= target.height - margin
        ) return null;

        let gxx = 0;
        let gxy = 0;
        let gyy = 0;
        let bx = 0;
        let by = 0;
        let residual = 0;
        let samples = 0;
        for (let offsetY = -lkWindowRadius; offsetY <= lkWindowRadius; offsetY++) {
            for (let offsetX = -lkWindowRadius; offsetX <= lkWindowRadius; offsetX++) {
                const sourceValue = sampleBilinear(
                    source.data, source.width, source.height,
                    sourcePoint.x + offsetX, sourcePoint.y + offsetY
                );
                const currentX = targetX + offsetX;
                const currentY = targetY + offsetY;
                const targetValue = sampleBilinear(target.data, target.width, target.height, currentX, currentY);
                const gx = 0.5 * (
                    sampleBilinear(target.data, target.width, target.height, currentX + 1, currentY) -
                    sampleBilinear(target.data, target.width, target.height, currentX - 1, currentY)
                );
                const gy = 0.5 * (
                    sampleBilinear(target.data, target.width, target.height, currentX, currentY + 1) -
                    sampleBilinear(target.data, target.width, target.height, currentX, currentY - 1)
                );
                if (![sourceValue, targetValue, gx, gy].every(Number.isFinite)) return null;
                const error = sourceValue - targetValue;
                gxx += gx * gx;
                gxy += gx * gy;
                gyy += gy * gy;
                bx += gx * error;
                by += gy * error;
                residual += Math.abs(error);
                samples++;
            }
        }
        const determinant = gxx * gyy - gxy * gxy;
        if (determinant < 1e-3) return null;
        const deltaX = (gyy * bx - gxy * by) / determinant;
        const deltaY = (gxx * by - gxy * bx) / determinant;
        if (!Number.isFinite(deltaX) || !Number.isFinite(deltaY)) return null;
        targetX += clamp(deltaX, -2.0, 2.0);
        targetY += clamp(deltaY, -2.0, 2.0);
        if (Math.hypot(deltaX, deltaY) < 0.025) {
            return { x: targetX, y: targetY, residual: residual / Math.max(1, samples) };
        }
    }
    return { x: targetX, y: targetY, residual: 0 };
}

function trackPoint(sourcePyramid, targetPyramid, point, initialPoint) {
    let tracked = null;
    for (let level = pyramidLevels - 1; level >= 0; level--) {
        const levelScale = 2 ** level;
        const sourcePoint = { x: point.x / levelScale, y: point.y / levelScale };
        const initial = tracked
            ? { x: tracked.x * 2, y: tracked.y * 2 }
            : { x: initialPoint.x / levelScale, y: initialPoint.y / levelScale };
        tracked = trackPointAtLevel(sourcePyramid[level], targetPyramid[level], sourcePoint, initial);
        if (!tracked) return null;
    }
    return tracked;
}

function trackFeatures(previous, current, features, predictedFeatures) {
    const previousGood = [];
    const currentGood = [];
    for (let index = 0; index < features.length; index++) {
        const feature = features[index];
        const predicted = predictedFeatures && predictedFeatures[index] ? predictedFeatures[index] : feature;
        const forward = trackPoint(previous, current, feature, predicted);
        if (!forward || forward.residual > 42) continue;
        const backward = trackPoint(current, previous, forward, feature);
        if (!backward) continue;
        if (Math.hypot(backward.x - feature.x, backward.y - feature.y) > forwardBackwardLimitPx) continue;
        previousGood.push(feature);
        currentGood.push({ x: forward.x, y: forward.y });
    }
    return { previous: previousGood, current: currentGood };
}

function multiplyMatrix3(left, right) {
    const output = new Array(9).fill(0);
    for (let row = 0; row < 3; row++) {
        for (let column = 0; column < 3; column++) {
            for (let k = 0; k < 3; k++) output[row * 3 + column] += left[row * 3 + k] * right[k * 3 + column];
        }
    }
    return output;
}

function transposeMatrix3(matrix) {
    return [matrix[0], matrix[3], matrix[6], matrix[1], matrix[4], matrix[7], matrix[2], matrix[5], matrix[8]];
}

function rotationMotion(previousRotation, currentRotation, dt) {
    if (!previousRotation || !currentRotation || dt <= 0) {
        return { angle: 0, rate: 0, axis: [0, 0, 0], yawDominant: false };
    }
    const relative = multiplyMatrix3(transposeMatrix3(currentRotation), previousRotation);
    const cosine = clamp((relative[0] + relative[4] + relative[8] - 1) / 2, -1, 1);
    const angle = Math.acos(cosine);
    const sine = Math.sin(angle);
    let axis = [0, 0, 0];
    if (Math.abs(sine) > 1e-6) {
        axis = [
            (relative[7] - relative[5]) / (2 * sine),
            (relative[2] - relative[6]) / (2 * sine),
            (relative[3] - relative[1]) / (2 * sine),
        ];
        const norm = Math.hypot(axis[0], axis[1], axis[2]);
        if (norm > 1e-6) axis = axis.map((value) => value / norm);
    }
    // Viewer-local +Y is the yaw axis.  Pitch (+X) and roll (+Z) produce
    // perspective-varying flow and must keep point-wise compensation.
    const yawDominant = Math.abs(axis[1]) >= Math.max(Math.abs(axis[0]), Math.abs(axis[2]));
    return { angle: angle, rate: angle / dt, axis: axis, yawDominant: yawDominant };
}

function predictRotationPoints(previousPoints, previousRotation, currentRotation, focal) {
    if (!previousRotation || !currentRotation) return previousPoints.map((point) => ({ x: point.x, y: point.y }));
    const relative = multiplyMatrix3(transposeMatrix3(currentRotation), previousRotation);
    const centerX = width / 2;
    const centerY = height / 2;
    return previousPoints.map((point) => {
        // WebXR camera axes are +X right, +Y up, +Z back, while image Y grows
        // downward.  Keep that sign difference explicit when projecting the
        // system-attitude-predicted rotational flow.
        const ray = [(point.x - centerX) / focal, -(point.y - centerY) / focal, -1];
        const rotated = [
            relative[0] * ray[0] + relative[1] * ray[1] + relative[2] * ray[2],
            relative[3] * ray[0] + relative[4] * ray[1] + relative[5] * ray[2],
            relative[6] * ray[0] + relative[7] * ray[1] + relative[8] * ray[2],
        ];
        if (Math.abs(rotated[2]) < 1e-5) return { x: point.x, y: point.y };
        return {
            x: centerX - focal * rotated[0] / rotated[2],
            y: centerY + focal * rotated[1] / rotated[2],
        };
    });
}

function rotationCompensatedPoints(previousPoints, currentPoints, predictedPoints) {
    return previousPoints.map((point, index) => ({
        x: point.x + currentPoints[index].x - predictedPoints[index].x,
        y: point.y + currentPoints[index].y - predictedPoints[index].y,
    }));
}

function median(values) {
    if (!values.length) return 0;
    const sorted = values.slice().sort((left, right) => left - right);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle] : 0.5 * (sorted[middle - 1] + sorted[middle]);
}

function focalScore(previousPoints, currentPoints, previousRotation, currentRotation, focal) {
    const predicted = predictRotationPoints(previousPoints, previousRotation, currentRotation, focal);
    const residualX = currentPoints.map((point, index) => point.x - predicted[index].x);
    const residualY = currentPoints.map((point, index) => point.y - predicted[index].y);
    const centerX = median(residualX);
    const centerY = median(residualY);
    const residuals = residualX.map((value, index) => Math.hypot(value - centerX, residualY[index] - centerY));
    return median(residuals);
}

function refineFocalLength(previousPoints, currentPoints, previousRotation, currentRotation, drawWidth) {
    if (!previousRotation || !currentRotation || previousPoints.length < minReliableFeatures) {
        return { focal: focalLengthPx, confidence: 0 };
    }
    const relative = multiplyMatrix3(transposeMatrix3(currentRotation), previousRotation);
    const perspectiveStrength = Math.hypot(relative[2], relative[5], relative[6], relative[7]);
    if (perspectiveStrength < 0.008) return { focal: focalLengthPx, confidence: focalConfidence };

    const lower = drawWidth * 0.55;
    const upper = drawWidth * 2.0;
    const coarseStep = (upper - lower) / 8;
    let bestFocal = clamp(focalLengthPx, lower, upper);
    let bestScore = focalScore(previousPoints, currentPoints, previousRotation, currentRotation, bestFocal);
    for (let index = 0; index <= 8; index++) {
        const candidate = lower + index * coarseStep;
        const score = focalScore(previousPoints, currentPoints, previousRotation, currentRotation, candidate);
        if (score < bestScore) {
            bestScore = score;
            bestFocal = candidate;
        }
    }
    const refineStep = coarseStep / 4;
    for (let offset = -2; offset <= 2; offset++) {
        const candidate = clamp(bestFocal + offset * refineStep, lower, upper);
        const score = focalScore(previousPoints, currentPoints, previousRotation, currentRotation, candidate);
        if (score < bestScore) {
            bestScore = score;
            bestFocal = candidate;
        }
    }
    const confidence = clamp(1 / (1 + bestScore / 3.0), 0, 1);
    return { focal: bestFocal, confidence: confidence };
}

function similarityFromPair(previousPoints, currentPoints, first, second) {
    const previousDx = previousPoints[second].x - previousPoints[first].x;
    const previousDy = previousPoints[second].y - previousPoints[first].y;
    const denominator = previousDx * previousDx + previousDy * previousDy;
    if (denominator < 4) return null;
    const currentDx = currentPoints[second].x - currentPoints[first].x;
    const currentDy = currentPoints[second].y - currentPoints[first].y;
    const a = (previousDx * currentDx + previousDy * currentDy) / denominator;
    const b = (previousDx * currentDy - previousDy * currentDx) / denominator;
    return {
        a: a,
        b: b,
        tx: currentPoints[first].x - a * previousPoints[first].x + b * previousPoints[first].y,
        ty: currentPoints[first].y - b * previousPoints[first].x - a * previousPoints[first].y,
    };
}

function similarityResidual(model, previousPoint, currentPoint) {
    const predictedX = model.a * previousPoint.x - model.b * previousPoint.y + model.tx;
    const predictedY = model.b * previousPoint.x + model.a * previousPoint.y + model.ty;
    return Math.hypot(currentPoint.x - predictedX, currentPoint.y - predictedY);
}

function fitSimilarity(previousPoints, currentPoints, indices) {
    let previousCenterX = 0;
    let previousCenterY = 0;
    let currentCenterX = 0;
    let currentCenterY = 0;
    for (const index of indices) {
        previousCenterX += previousPoints[index].x;
        previousCenterY += previousPoints[index].y;
        currentCenterX += currentPoints[index].x;
        currentCenterY += currentPoints[index].y;
    }
    previousCenterX /= indices.length;
    previousCenterY /= indices.length;
    currentCenterX /= indices.length;
    currentCenterY /= indices.length;
    let dot = 0;
    let cross = 0;
    let norm = 0;
    for (const index of indices) {
        const px = previousPoints[index].x - previousCenterX;
        const py = previousPoints[index].y - previousCenterY;
        const cx = currentPoints[index].x - currentCenterX;
        const cy = currentPoints[index].y - currentCenterY;
        dot += px * cx + py * cy;
        cross += px * cy - py * cx;
        norm += px * px + py * py;
    }
    const a = norm > 1e-6 ? dot / norm : 1;
    const b = norm > 1e-6 ? cross / norm : 0;
    return {
        a: a,
        b: b,
        tx: currentCenterX - a * previousCenterX + b * previousCenterY,
        ty: currentCenterY - b * previousCenterX - a * previousCenterY,
    };
}

function summarizeSimilarity(model, rms, indices) {
    const centerX = width / 2;
    const centerY = height / 2;
    const mappedCenterX = model.a * centerX - model.b * centerY + model.tx;
    const mappedCenterY = model.b * centerX + model.a * centerY + model.ty;
    return {
        x: mappedCenterX - centerX,
        y: mappedCenterY - centerY,
        scale: Math.hypot(model.a, model.b) - 1,
        rms: rms,
        indices: indices,
    };
}

function robustSimilarity(previousPoints, currentPoints) {
    if (previousPoints.length < 2) return null;
    let bestIndices = [];
    for (let iteration = 0; iteration < ransacIterations; iteration++) {
        const first = iteration % previousPoints.length;
        const second = (iteration * 17 + 7) % previousPoints.length;
        if (first === second) continue;
        const candidate = similarityFromPair(previousPoints, currentPoints, first, second);
        if (!candidate) continue;
        const indices = [];
        for (let index = 0; index < previousPoints.length; index++) {
            if (similarityResidual(candidate, previousPoints[index], currentPoints[index]) <= ransacThresholdPx) {
                indices.push(index);
            }
        }
        if (indices.length > bestIndices.length) bestIndices = indices;
    }
    if (bestIndices.length < 3) return null;
    const model = fitSimilarity(previousPoints, currentPoints, bestIndices);
    let squaredError = 0;
    for (const index of bestIndices) {
        const residual = similarityResidual(model, previousPoints[index], currentPoints[index]);
        squaredError += residual * residual;
    }
    return summarizeSimilarity(model, Math.sqrt(squaredError / bestIndices.length), bestIndices);
}

function rotationModelResidualFit(previousPoints, predictedPoints, observedFit) {
    // A small camera/attitude timing or focal mismatch can destroy consensus after
    // point-wise yaw compensation.  Fit observed motion first, then subtract
    // the predicted rotation model so common lateral translation survives.
    if (!observedFit) return null;
    const predictedModel = fitSimilarity(previousPoints, predictedPoints, observedFit.indices);
    const predictedFit = summarizeSimilarity(predictedModel, 0, observedFit.indices);
    return {
        x: observedFit.x - predictedFit.x,
        y: observedFit.y - predictedFit.y,
        scale: observedFit.scale - predictedFit.scale,
        rms: observedFit.rms,
        indices: observedFit.indices,
    };
}

function resetTracking() {
    previousPyramid = null;
    previousFeatures = [];
    previousOrientation = null;
    previousTimestamp = 0;
}

function processFrame(bitmap, timestamp, orientation) {
    context.fillStyle = "black";
    context.fillRect(0, 0, width, height);
    const scale = Math.min(width / bitmap.width, height / bitmap.height);
    const drawWidth = Math.round(bitmap.width * scale);
    const drawHeight = Math.round(bitmap.height * scale);
    const offsetX = Math.round((width - drawWidth) / 2);
    const offsetY = Math.round((height - drawHeight) / 2);
    context.drawImage(bitmap, offsetX, offsetY, drawWidth, drawHeight);
    bitmap.close();
    if (focalLengthPx <= 0) {
        const initialHorizontalFovRad = 65 * Math.PI / 180;
        focalLengthPx = drawWidth / (2 * Math.tan(initialHorizontalFovRad / 2));
    }
    const currentGray = grayscale(context.getImageData(0, 0, width, height));
    const currentPyramid = buildPyramid(currentGray);
    const currentOrientation = Array.isArray(orientation) && orientation.length === 9 ? orientation.slice() : null;

    if (!previousPyramid || timestamp - previousTimestamp > 200) {
        previousPyramid = currentPyramid;
        previousFeatures = detectFeatures(currentGray);
        previousOrientation = currentOrientation;
        previousTimestamp = timestamp;
        return {
            delta: [0, 0, 0],
            quality: 0,
            accepted: false,
            translationSuppressed: false,
            translationGain: 0,
            rotationRateRadS: 0,
            rotationCompensationConfidence: 0,
            yawModelUsed: false,
            focalLengthPx: focalLengthPx,
            reliableCount: 0,
            features: previousFeatures,
        };
    }

    const dt = Math.max(0.001, (timestamp - previousTimestamp) / 1000);
    const rotation = rotationMotion(previousOrientation, currentOrientation, dt);
    const rotationActive = rotation.angle >= rotationFusionMinRad && rotation.rate >= rotationFusionMinRadS;
    const initialPrediction = predictRotationPoints(
        previousFeatures, previousOrientation, currentOrientation, focalLengthPx
    );
    const tracked = trackFeatures(previousPyramid, currentPyramid, previousFeatures, initialPrediction);
    if (rotationActive && tracked.current.length >= minReliableFeatures) {
        const focalEstimate = refineFocalLength(
            tracked.previous,
            tracked.current,
            previousOrientation,
            currentOrientation,
            drawWidth
        );
        if (focalEstimate.confidence >= 0.25) {
            const focalAlpha = 0.08 + 0.12 * focalEstimate.confidence;
            focalLengthPx = (1 - focalAlpha) * focalLengthPx + focalAlpha * focalEstimate.focal;
            focalConfidence = (1 - focalAlpha) * focalConfidence + focalAlpha * focalEstimate.confidence;
        }
    }
    const predicted = predictRotationPoints(
        tracked.previous, previousOrientation, currentOrientation, focalLengthPx
    );
    const rawFit = robustSimilarity(tracked.previous, tracked.current);
    const compensated = rotationCompensatedPoints(tracked.previous, tracked.current, predicted);
    const pointCompensatedFit = robustSimilarity(tracked.previous, compensated);
    const modelResidualFit = rotationActive && rotation.yawDominant
        ? rotationModelResidualFit(tracked.previous, predicted, rawFit)
        : null;
    // Gate attitude subtraction so harmless heading noise cannot cancel
    // ordinary lateral optical flow while the phone is translating.
    let fit = rotationActive ? pointCompensatedFit : rawFit;
    let yawModelUsed = false;
    // During yaw, model-level subtraction preserves common lateral translation.
    // Pitch/roll stay on point-wise perspective compensation: a similarity
    // model cannot represent their non-uniform image motion and otherwise turns
    // pure camera tilt into a large false vertical translation.
    if (modelResidualFit && modelResidualFit.indices.length >= minReliableFeatures) {
        fit = modelResidualFit;
        yawModelUsed = true;
    }
    let delta = [0, 0, 0];
    let quality = 0;
    let accepted = false;
    let translationGain = 1;
    let rotationCompensationConfidence = 1;
    if (rotationActive && tracked.current.length >= minReliableFeatures) {
        const compensationResidual = focalScore(
            tracked.previous,
            tracked.current,
            previousOrientation,
            currentOrientation,
            focalLengthPx
        );
        rotationCompensationConfidence = clamp(1 / (1 + compensationResidual / 3.0), 0, 1);
    }
    let reliableFeatures = tracked.current;
    if (fit) {
        reliableFeatures = fit.indices.map((index) => tracked.current[index]);
        const trackRatio = tracked.current.length / Math.max(1, previousFeatures.length);
        const inlierRatio = reliableFeatures.length / Math.max(1, tracked.current.length);
        const featureCoverage = Math.min(1, reliableFeatures.length / 45);
        const residualConfidence = 1 / (1 + fit.rms / 2.5);
        // Coverage is the main confidence term.  Ratios and residuals modulate
        // it instead of being multiplied directly; direct multiplication made
        // ordinary parallax collapse an otherwise usable track to "poor".
        quality = clamp(
            featureCoverage *
            (0.50 + 0.50 * inlierRatio) *
            (0.55 + 0.45 * residualConfidence) *
            (0.75 + 0.25 * trackRatio),
            0,
            1
        );
        accepted = reliableFeatures.length >= minReliableFeatures && quality >= 0.20;
        if (accepted) {
            if (rotationActive) {
                const fusedConfidence = 0.70 * quality + 0.30 * rotationCompensationConfidence;
                const gainFloor = rotation.yawDominant
                    ? yawRotationTranslationGainFloor
                    : tiltRotationTranslationGainFloor;
                translationGain = rotationTranslationGain(fusedConfidence, gainFloor);
            }
            let dxNormalized = fit.x / Math.max(1, drawWidth);
            let dyNormalized = fit.y / Math.max(1, drawWidth);
            let depthScale = fit.scale;
            const lateralDeadzone = lateralDeadzoneNormalized +
                (rotationActive ? Math.min(0.0005, rotation.rate * 0.00035) : 0);
            const dynamicDepthDeadzone = depthDeadzone +
                (rotationActive ? Math.min(0.001, rotation.rate * 0.0005) : 0);
            if (Math.abs(dxNormalized) < lateralDeadzone) dxNormalized = 0;
            if (Math.abs(dyNormalized) < lateralDeadzone) dyNormalized = 0;
            if (Math.abs(depthScale) < dynamicDepthDeadzone) depthScale = 0;
            delta = [
                -dxNormalized * lateralGainMeters,
                dyNormalized * lateralGainMeters,
                -depthScale * depthGainMeters,
            ].map((value) => value * translationGain);
            delta[2] = clamp(delta[2], -maxFrameDepthM, maxFrameDepthM);
            const magnitude = Math.hypot(delta[0], delta[1], delta[2]);
            const frameLimit = rotationActive ? maxFrameTranslationM * 0.85 : maxFrameTranslationM;
            if (magnitude > frameLimit) {
                const ratio = frameLimit / magnitude;
                delta = delta.map((value) => value * ratio);
            }
        }
    }
    if (!accepted && rotationActive) translationGain = 0;
    const translationSuppressed = rotationActive && translationGain <= 0.05;

    // Keep observed image points for the next frame.  If tracking becomes
    // sparse, reacquire a fresh grid while reporting a held pose to the page.
    const nextFeatures = reliableFeatures.length >= 55 ? reliableFeatures : detectFeatures(currentGray);
    previousPyramid = currentPyramid;
    previousFeatures = nextFeatures;
    previousOrientation = currentOrientation;
    previousTimestamp = timestamp;
    return {
        delta: delta,
        quality: quality,
        accepted: accepted,
        translationSuppressed: translationSuppressed,
        translationGain: translationGain,
        rotationRateRadS: rotation.rate,
        rotationAxis: rotation.axis,
        rotationYawDominant: rotation.yawDominant,
        rotationCompensationConfidence: rotationCompensationConfidence,
        yawModelUsed: yawModelUsed,
        focalLengthPx: focalLengthPx,
        frameOrientation: currentOrientation,
        reliableCount: fit ? reliableFeatures.length : 0,
        features: nextFeatures,
    };
}

self.onmessage = (event) => {
    if (event.data.type === "init") {
        width = event.data.width;
        height = event.data.height;
        canvas = new OffscreenCanvas(width, height);
        context = canvas.getContext("2d", { willReadFrequently: true });
        resetTracking();
        self.postMessage({ type: "ready" });
        return;
    }
    if (event.data.type === "reset") {
        resetTracking();
        return;
    }
    if (event.data.type !== "frame" || !context) return;
    try {
        const startedAt = performance.now();
        const result = processFrame(event.data.bitmap, event.data.timestamp, event.data.orientation);
        self.postMessage({
            type: "result",
            ...result,
            timestamp: event.data.timestamp,
            processingMs: performance.now() - startedAt,
        });
    } catch (error) {
        if (event.data.bitmap && event.data.bitmap.close) event.data.bitmap.close();
        self.postMessage({ type: "error", message: String(error) });
    }
};
