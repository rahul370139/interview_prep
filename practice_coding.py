#1. two number adds up to target

def two_num(nums: list[int], target: int) -> list[int]:
    seen = {}
    for i, num in enumerate(nums):
        find = target - nums[i]
        if find in seen:
            return [seen[find], i]
        seen[num] = i
    return []

print(two_num([2, 7, 11, 15], 9))
print(two_num([3, 2, 4], 6))
print(two_num([3, 3], 6))
print(two_num([3, 2, 4], 6))
print(two_num([3, 2, 4], 6))


