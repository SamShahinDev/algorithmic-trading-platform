"""
Async utilities for safe awaiting and coroutine handling
"""
import inspect
import asyncio
import logging

logger = logging.getLogger(__name__)

async def await_if_coro(maybe_coro):
    """
    Safely await a value only if it's a coroutine or awaitable.
    Returns the value directly if it's neither.
    
    Args:
        maybe_coro: Value that might be a coroutine, awaitable, or plain value
        
    Returns:
        The awaited result or the value itself
    """
    if maybe_coro is None:
        return None
        
    if inspect.iscoroutine(maybe_coro):
        return await maybe_coro
    if inspect.isawaitable(maybe_coro):
        return await maybe_coro
    
    # Log once on first non-awaitable handling
    if not hasattr(await_if_coro, '_logged'):
        await_if_coro._logged = True
        logger.info("AWAIT_SAFE wrapped_call=%s type=%s", 
                   getattr(maybe_coro, '__name__', 'unknown'), 
                   type(maybe_coro).__name__)
    
    return maybe_coro  # plain value or None