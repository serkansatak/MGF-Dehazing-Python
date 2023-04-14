from Utils.ConfigUtil import parseParams
from Utils.ProcessorMGF import ProcessorMGF

def main():
    args = parseParams()
    
    print(args)
    
    processor = ProcessorMGF(configArgs=args, src=args.input, out=args.output)
    
    processor.initializeMGF()
    processor.operator = processor.operationMGF
    
    processor.printAttributes()
    
    print("Starting process...")
    processor.processVideo()
    
    
if __name__ == "__main__":
    main()
    exit()