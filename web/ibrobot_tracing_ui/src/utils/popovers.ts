const SELECTOR = "details[data-popover][open]";

function close(popover: HTMLDetailsElement, restoreFocus: boolean): void {
  popover.open = false;
  if (restoreFocus) popover.querySelector<HTMLElement>(":scope > summary")?.focus({ preventScroll: true });
}

/** Floating disclosure first; inline disclosure (for example raw fields) stays open. */
export function dismissTopPopover(): boolean {
  const open = [...document.querySelectorAll<HTMLDetailsElement>(SELECTOR)];
  const focused = document.activeElement?.closest<HTMLDetailsElement>(SELECTOR);
  const top = focused ?? open[open.length - 1];
  if (!top) return false;
  close(top, true);
  return true;
}

export function dismissOutsidePopovers(event: PointerEvent): void {
  if (!(event.target instanceof Node)) return;
  for (const popover of document.querySelectorAll<HTMLDetailsElement>(SELECTOR)) {
    if (!popover.contains(event.target)) close(popover, popover.contains(document.activeElement));
  }
}

export function keepOnePopover(event: Event): void {
  const current = event.target;
  if (!(current instanceof HTMLDetailsElement) || !current.matches(SELECTOR)) return;
  for (const popover of document.querySelectorAll<HTMLDetailsElement>(SELECTOR)) {
    if (popover !== current) close(popover, false);
  }
}
