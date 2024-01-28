#include <CL/sycl.hpp>

/**
 * A queue is created with a device
 * Queue submission creates a handler
 * The handler is passed to accessors
 * Memory management is implicit after this point
 * 
 * I need my data (such as the CSR) to know it lives inside a queue
 * and to update its accessors accordingly 
 * 
 * Basically
 * if (inside_queue)
 *      set data accessors = queue accessors with handler
 * else
 *      set accessors = cpu accessors
 * 
 * A queue can potentially be very long-lived, with many submissions
 * 
 * Therefore, it is impossible for something to know about which device is operative
 * prior to queues being created, since it is the handler, created by the queue
 * that passes this information to the device
 * 
 * Logically, then, I need to create queues first, and anything that
 * can be accessed on devices needs to be told about those
 * queues afterward
*/


class QueueManager
{
public:

private:
};