from Utils.ConfigUtil import parseParams
from Utils.ProcessorMGF import ProcessorMGF

def main():
    args = parseParams()
    
    print(args)
    
    processor = ProcessorMGF(configArgs=args, src=args.input, out=args.output)    
    processor.operator = processor.operationMGF
    
    processor.printAttributes()
    
    print("Starting process...")
    processor.processSequence()
    processor.printAttributes
    
    
if __name__ == "__main__":
    main()
    exit()